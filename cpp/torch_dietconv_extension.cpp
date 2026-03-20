#include <torch/extension.h>

#include <ATen/Parallel.h>
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

// This file implements the CPU-only PyTorch extension used by the torch
// benchmarks. The important distinction is:
//
// - DietConv v1 keeps the "one output row at a time" strip-buffer schedule
//   from the poster. Each output row copies an entire Fh x padded_width strip
//   for every input channel, then runs one GEMM per kernel column.
// - DietConv v2 keeps the same per-kernel-column GEMM structure but tiles the
//   output width. That reduces the peak temporary footprint and gives us a
//   smaller window to keep warm in cache.
//
// The code is intentionally explicit about data movement. Most of the runtime
// cost in these kernels is not arithmetic; it is deciding what to pack, when to
// repack it, and how large a GEMM-backed lowering view should be.

int64_t resolve_tile_out_width(int64_t out_w, int64_t stride, int64_t tile_out_width) {
    if (tile_out_width > 0) {
        return std::min(tile_out_width, out_w);
    }
    const int64_t heuristic = stride == 1 ? 128 : 32;
    return std::min(heuristic, out_w);
}

void validate_input_and_weight(const torch::Tensor& input, const torch::Tensor& weight) {
    // The extension only implements the narrow CPU inference subset used by the
    // benchmarks: dense float32 NCHW input and dense float32 OIHW weights.
    TORCH_CHECK(input.device().is_cpu(), "DietConv extension only supports CPU tensors.");
    TORCH_CHECK(weight.device().is_cpu(), "DietConv extension only supports CPU tensors.");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "DietConv extension expects float32 input.");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "DietConv extension expects float32 weight.");
    TORCH_CHECK(input.dim() == 4, "Expected input with shape (N, C, H, W).");
    TORCH_CHECK(weight.dim() == 4, "Expected weight with shape (K, C, Fh, Fw).");
    TORCH_CHECK(input.size(1) == weight.size(1), "Input channel count and weight channel count must match.");
}

void validate_input_and_packed_weight(const torch::Tensor& input, const torch::Tensor& packed_weight) {
    // Packed weights store one matrix per kernel-column:
    //
    //   packed_weight[kernel_x] -> [out_channels, channels * kernel_height]
    //
    // That layout makes each kernel-column contribution a single GEMM.
    TORCH_CHECK(input.device().is_cpu(), "DietConv extension only supports CPU tensors.");
    TORCH_CHECK(packed_weight.device().is_cpu(), "DietConv extension only supports CPU tensors.");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "DietConv extension expects float32 input.");
    TORCH_CHECK(packed_weight.scalar_type() == at::kFloat, "DietConv extension expects float32 packed weights.");
    TORCH_CHECK(input.dim() == 4, "Expected input with shape (N, C, H, W).");
    TORCH_CHECK(packed_weight.dim() == 3, "Expected packed weight with shape (Fw, K, C*Fh).");
}

torch::Tensor make_padded_input(const torch::Tensor& input, int64_t padding) {
    // Padding is materialized once up front so the hot loops do not need bounds
    // checks or branchy edge handling.
    if (padding == 0) {
        return input.contiguous();
    }
    auto padded = torch::zeros(
        {
            input.size(0),
            input.size(1),
            input.size(2) + 2 * padding,
            input.size(3) + 2 * padding,
        },
        input.options()
    );
    padded.slice(2, padding, padding + input.size(2)).slice(3, padding, padding + input.size(3)).copy_(input);
    return padded;
}

torch::Tensor normalize_bias(const c10::optional<torch::Tensor>& bias, int64_t out_channels) {
    if (!bias.has_value()) {
        return torch::Tensor();
    }
    auto bias_tensor = bias.value().contiguous();
    TORCH_CHECK(bias_tensor.device().is_cpu(), "Bias must be on CPU.");
    TORCH_CHECK(bias_tensor.scalar_type() == at::kFloat, "Bias must be float32.");
    TORCH_CHECK(bias_tensor.dim() == 1, "Bias must have shape (K,).");
    TORCH_CHECK(bias_tensor.size(0) == out_channels, "Bias output channels must match output.");
    return bias_tensor;
}

torch::Tensor prepack_weight(torch::Tensor weight) {
    // Reorder weights from [K, C, Fh, Fw] to [Fw, K, C * Fh].
    //
    // The poster groups work by kernel-column. Doing the same here lets the hot
    // path issue:
    //
    //   row_out += packed_weight[kernel_x] @ lowering_slice_for_kernel_x
    //
    // instead of rebuilding a new view of the weights every iteration.
    validate_input_and_weight(torch::empty({1, weight.size(1), 1, 1}, weight.options()), weight);
    auto contiguous = weight.contiguous();
    const int64_t out_channels = contiguous.size(0);
    const int64_t channels = contiguous.size(1);
    const int64_t kernel_height = contiguous.size(2);
    const int64_t kernel_width = contiguous.size(3);
    const int64_t slice_inner = channels * kernel_height;

    auto packed = torch::empty({kernel_width, out_channels, slice_inner}, contiguous.options());
    const float* weight_ptr = contiguous.data_ptr<float>();
    float* packed_ptr = packed.data_ptr<float>();

    for (int64_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
        for (int64_t out_channel = 0; out_channel < out_channels; ++out_channel) {
            for (int64_t channel = 0; channel < channels; ++channel) {
                for (int64_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
                    const int64_t slice_index = channel * kernel_height + kernel_y;
                    const int64_t packed_index =
                        ((kernel_x * out_channels + out_channel) * slice_inner) + slice_index;
                    const int64_t weight_index =
                        (((out_channel * channels + channel) * kernel_height + kernel_y) * kernel_width) + kernel_x;
                    packed_ptr[packed_index] = weight_ptr[weight_index];
                }
            }
        }
    }
    return packed;
}

torch::Tensor dietconv_v1_prepacked_forward(
    torch::Tensor input,
    torch::Tensor packed_weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding
) {
    // v1 is the direct strip-buffer schedule:
    //
    // 1. Materialize one Fh x padded_width strip per input channel for one
    //    output row.
    // 2. For each kernel-column, view or repack the relevant lowering slice.
    // 3. Accumulate one GEMM into row_out.
    //
    // The work is parallelized over (batch, out_y) pairs so each worker owns an
    // entire output row and can keep its scratch buffers thread-local.
    validate_input_and_packed_weight(input, packed_weight);
    TORCH_CHECK(stride > 0, "stride must be positive.");
    TORCH_CHECK(padding >= 0, "padding must be non-negative.");

    auto x = input.contiguous();
    auto packed = packed_weight.contiguous();
    const int64_t batch_size = x.size(0);
    const int64_t channels = x.size(1);
    const int64_t kernel_width = packed.size(0);
    const int64_t out_channels = packed.size(1);
    const int64_t slice_inner = packed.size(2);
    TORCH_CHECK(slice_inner % channels == 0, "Packed weight slice size must be divisible by channel count.");
    const int64_t kernel_height = slice_inner / channels;

    auto padded = make_padded_input(x, padding);
    const int64_t padded_height = padded.size(2);
    const int64_t padded_width = padded.size(3);
    const int64_t out_h = (padded_height - kernel_height) / stride + 1;
    const int64_t out_w = (padded_width - kernel_width) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());
    auto bias_tensor = normalize_bias(bias, out_channels);
    const float* bias_ptr = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const float* padded_ptr = padded.data_ptr<float>();
    const float* packed_ptr = packed.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    const int64_t padded_batch_stride = channels * padded_height * padded_width;
    const int64_t output_batch_stride = out_channels * out_h * out_w;

    at::parallel_for(0, batch_size * out_h, 1, [&](int64_t begin, int64_t end) {
        // temp:  [slice_inner, padded_width]
        // strip: [slice_inner, out_w]    only needed when stride > 1
        // row_out: [out_channels, out_w]
        std::vector<float> temp(static_cast<std::size_t>(slice_inner) * padded_width, 0.0f);
        std::vector<float> row_out(static_cast<std::size_t>(out_channels) * out_w, 0.0f);
        std::vector<float> strip(static_cast<std::size_t>(slice_inner) * out_w, 0.0f);
        for (int64_t index = begin; index < end; ++index) {
            const int64_t batch = index / out_h;
            const int64_t out_y = index % out_h;
            const int64_t row_start = out_y * stride;
            const float* batch_padded = padded_ptr + batch * padded_batch_stride;
            float* batch_output = output_ptr + batch * output_batch_stride;

            // Copy the full vertical strip needed for this output row. This is
            // the central v1 memory tradeoff: much smaller than full im2col, but
            // still wide enough to cover the entire output row in one go.
            for (int64_t row = 0; row < slice_inner; ++row) {
                const int64_t channel = row / kernel_height;
                const int64_t kernel_y = row % kernel_height;
                const float* src = batch_padded + ((channel * padded_height + row_start + kernel_y) * padded_width);
                std::memcpy(
                    temp.data() + static_cast<std::size_t>(row) * padded_width,
                    src,
                    static_cast<std::size_t>(padded_width) * sizeof(float)
                );
            }

            std::fill(row_out.begin(), row_out.end(), 0.0f);
            for (int64_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
                const float* lowering_matrix = nullptr;
                int lowering_leading_dim = static_cast<int>(out_w);
                if (stride == 1) {
                    // With stride 1 the lowering slice is already contiguous
                    // enough for GEMM to consume directly from the strip buffer.
                    lowering_matrix = temp.data() + kernel_x;
                    lowering_leading_dim = static_cast<int>(padded_width);
                } else {
                    // Strided cases need a compact gather into strip so GEMM sees
                    // a dense [slice_inner, out_w] matrix.
                    for (int64_t row = 0; row < slice_inner; ++row) {
                        const float* temp_row = temp.data() + static_cast<std::size_t>(row) * padded_width + kernel_x;
                        float* strip_row = strip.data() + static_cast<std::size_t>(row) * out_w;
                        for (int64_t out_x = 0; out_x < out_w; ++out_x) {
                            strip_row[out_x] = temp_row[out_x * stride];
                        }
                    }
                    lowering_matrix = strip.data();
                }

                cblas_sgemm(
                    // [K, slice_inner] @ [slice_inner, out_w] -> [K, out_w]
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    static_cast<int>(out_channels),
                    static_cast<int>(out_w),
                    static_cast<int>(slice_inner),
                    1.0f,
                    packed_ptr + (kernel_x * out_channels * slice_inner),
                    static_cast<int>(slice_inner),
                    lowering_matrix,
                    lowering_leading_dim,
                    kernel_x == 0 ? 0.0f : 1.0f,
                    row_out.data(),
                    static_cast<int>(out_w)
                );
            }

            for (int64_t out_channel = 0; out_channel < out_channels; ++out_channel) {
                float* dst = batch_output + ((out_channel * out_h + out_y) * out_w);
                const float* src = row_out.data() + static_cast<std::size_t>(out_channel) * out_w;
                if (bias_ptr == nullptr) {
                    std::memcpy(dst, src, static_cast<std::size_t>(out_w) * sizeof(float));
                } else {
                    const float bias_value = bias_ptr[out_channel];
                    for (int64_t out_x = 0; out_x < out_w; ++out_x) {
                        dst[out_x] = src[out_x] + bias_value;
                    }
                }
            }
        }
    });

    return output;
}

torch::Tensor dietconv_v2_prepacked_forward(
    torch::Tensor input,
    torch::Tensor packed_weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t tile_out_width
) {
    // v2 keeps the same GEMM decomposition as v1 but shrinks the lowering
    // window from "whole output row" to "one output tile".
    //
    // The main benefits are:
    // - lower peak temporary footprint
    // - better locality on moderate-width feature maps
    // - a smaller working set per thread
    //
    // The cost is more tile-level packing work and a tile-width choice that
    // materially affects performance.
    validate_input_and_packed_weight(input, packed_weight);
    TORCH_CHECK(stride > 0, "stride must be positive.");
    TORCH_CHECK(padding >= 0, "padding must be non-negative.");

    auto x = input.contiguous();
    auto packed = packed_weight.contiguous();
    const int64_t batch_size = x.size(0);
    const int64_t channels = x.size(1);
    const int64_t kernel_width = packed.size(0);
    const int64_t out_channels = packed.size(1);
    const int64_t slice_inner = packed.size(2);
    TORCH_CHECK(slice_inner % channels == 0, "Packed weight slice size must be divisible by channel count.");
    const int64_t kernel_height = slice_inner / channels;

    auto padded = make_padded_input(x, padding);
    const int64_t padded_height = padded.size(2);
    const int64_t padded_width = padded.size(3);
    const int64_t out_h = (padded_height - kernel_height) / stride + 1;
    const int64_t out_w = (padded_width - kernel_width) / stride + 1;
    const int64_t resolved_tile_out_width = resolve_tile_out_width(out_w, stride, tile_out_width);
    const int64_t max_tile_input_width = (resolved_tile_out_width - 1) * stride + kernel_width;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());
    auto bias_tensor = normalize_bias(bias, out_channels);
    const float* bias_ptr = bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr;

    const float* padded_ptr = padded.data_ptr<float>();
    const float* packed_ptr = packed.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    const int64_t padded_batch_stride = channels * padded_height * padded_width;
    const int64_t output_batch_stride = out_channels * out_h * out_w;
    const bool stride_one = stride == 1;
    at::parallel_for(0, batch_size * out_h, 1, [&](int64_t begin, int64_t end) {
        // temp:    [slice_inner, max_tile_input_width]
        // tile_out:[out_channels, resolved_tile_out_width]
        // strip:   [slice_inner, resolved_tile_out_width] for strided cases
        std::vector<float> temp(static_cast<std::size_t>(slice_inner) * max_tile_input_width, 0.0f);
        std::vector<float> tile_out(static_cast<std::size_t>(out_channels) * resolved_tile_out_width, 0.0f);
        std::vector<float> strip(static_cast<std::size_t>(slice_inner) * resolved_tile_out_width, 0.0f);

        for (int64_t index = begin; index < end; ++index) {
            const int64_t batch = index / out_h;
            const int64_t out_y = index % out_h;
            const int64_t row_start = out_y * stride;
            const float* batch_padded = padded_ptr + batch * padded_batch_stride;
            float* batch_output = output_ptr + batch * output_batch_stride;

            for (int64_t tile_start = 0; tile_start < out_w; tile_start += resolved_tile_out_width) {
                const int64_t tile_width = std::min(resolved_tile_out_width, out_w - tile_start);
                const int64_t input_x = tile_start * stride;
                const int64_t tile_input_width = (tile_width - 1) * stride + kernel_width;

                // Copy only the tile-sized input window needed for this
                // (batch, out_y, tile_start) region.
                for (int64_t row = 0; row < slice_inner; ++row) {
                    const int64_t channel = row / kernel_height;
                    const int64_t kernel_y = row % kernel_height;
                    const float* src =
                        batch_padded + ((channel * padded_height + row_start + kernel_y) * padded_width) + input_x;
                    std::memcpy(
                        temp.data() + static_cast<std::size_t>(row) * max_tile_input_width,
                        src,
                        static_cast<std::size_t>(tile_input_width) * sizeof(float)
                    );
                }

                std::fill(tile_out.begin(), tile_out.end(), 0.0f);
                for (int64_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
                    const float* lowering_matrix = nullptr;
                    int lowering_leading_dim = static_cast<int>(resolved_tile_out_width);
                    if (stride_one) {
                        // For stride 1, each kernel-column can read directly from
                        // the packed temp window with only an x-offset.
                        lowering_matrix = temp.data() + kernel_x;
                        lowering_leading_dim = static_cast<int>(max_tile_input_width);
                    } else {
                        // For strided cases, compact the scattered columns into a
                        // dense matrix before GEMM.
                        for (int64_t row = 0; row < slice_inner; ++row) {
                            const float* temp_row = temp.data() + static_cast<std::size_t>(row) * max_tile_input_width + kernel_x;
                            float* strip_row = strip.data() + static_cast<std::size_t>(row) * resolved_tile_out_width;
                            for (int64_t out_x = 0; out_x < tile_width; ++out_x) {
                                strip_row[out_x] = temp_row[out_x * stride];
                            }
                        }
                        lowering_matrix = strip.data();
                    }

                    cblas_sgemm(
                        // [K, slice_inner] @ [slice_inner, tile_width]
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        static_cast<int>(out_channels),
                        static_cast<int>(tile_width),
                        static_cast<int>(slice_inner),
                        1.0f,
                        packed_ptr + (kernel_x * out_channels * slice_inner),
                        static_cast<int>(slice_inner),
                        lowering_matrix,
                        lowering_leading_dim,
                        kernel_x == 0 ? 0.0f : 1.0f,
                        tile_out.data(),
                        static_cast<int>(resolved_tile_out_width)
                    );
                }

                for (int64_t out_channel = 0; out_channel < out_channels; ++out_channel) {
                    float* dst = batch_output + ((out_channel * out_h + out_y) * out_w) + tile_start;
                    const float* src = tile_out.data() + static_cast<std::size_t>(out_channel) * resolved_tile_out_width;
                    if (bias_ptr == nullptr) {
                        std::memcpy(dst, src, static_cast<std::size_t>(tile_width) * sizeof(float));
                    } else {
                        const float bias_value = bias_ptr[out_channel];
                        for (int64_t out_x = 0; out_x < tile_width; ++out_x) {
                            dst[out_x] = src[out_x] + bias_value;
                        }
                    }
                }
            }
        }
    });

    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Only the prepacked forward variants are exported. Weight prepacking is the
    // core reusable systems trick here; it lets repeated forwards amortize the
    // weight-layout transform instead of paying it inside the timed path.
    m.def("prepack_weight", &prepack_weight, "Prepack DietConv weights (CPU)");
    m.def("dietconv_v1_prepacked_forward", &dietconv_v1_prepacked_forward, "DietConv v1 prepacked forward (CPU)");
    m.def("dietconv_v2_prepacked_forward", &dietconv_v2_prepacked_forward, "DietConv v2 prepacked forward (CPU)");
}
