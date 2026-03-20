#include <torch/extension.h>

#include <ATen/TensorIndexing.h>
#include <c10/util/Optional.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

using at::indexing::Slice;

namespace {

std::pair<int64_t, int64_t> to_pair(int64_t value) {
    return {value, value};
}

int64_t resolve_tile_out_width(int64_t out_w, int64_t stride, int64_t tile_out_width) {
    if (tile_out_width > 0) {
        return std::min(tile_out_width, out_w);
    }
    const int64_t heuristic = stride == 1 ? 128 : 32;
    return std::min(heuristic, out_w);
}

void validate_inputs(const torch::Tensor& input, const torch::Tensor& weight) {
    TORCH_CHECK(input.device().is_cpu(), "DietConv extension only supports CPU tensors.");
    TORCH_CHECK(weight.device().is_cpu(), "DietConv extension only supports CPU tensors.");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "DietConv extension expects float32 input.");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "DietConv extension expects float32 weight.");
    TORCH_CHECK(input.dim() == 4, "Expected input with shape (N, C, H, W).");
    TORCH_CHECK(weight.dim() == 4, "Expected weight with shape (K, C, Fh, Fw).");
    TORCH_CHECK(input.size(1) == weight.size(1), "Input channel count and weight channel count must match.");
}

torch::Tensor make_padded_input(const torch::Tensor& input, int64_t pad_h, int64_t pad_w) {
    if (pad_h == 0 && pad_w == 0) {
        return input.contiguous();
    }
    auto padded = torch::zeros(
        {
            input.size(0),
            input.size(1),
            input.size(2) + 2 * pad_h,
            input.size(3) + 2 * pad_w,
        },
        input.options()
    );
    padded.index_put_(
        {
            Slice(),
            Slice(),
            Slice(pad_h, pad_h + input.size(2)),
            Slice(pad_w, pad_w + input.size(3)),
        },
        input
    );
    return padded;
}

torch::Tensor add_bias_if_present(const torch::Tensor& output, const c10::optional<torch::Tensor>& bias) {
    if (!bias.has_value()) {
        return output;
    }
    auto bias_tensor = bias.value();
    TORCH_CHECK(bias_tensor.device().is_cpu(), "Bias must be on CPU.");
    TORCH_CHECK(bias_tensor.scalar_type() == at::kFloat, "Bias must be float32.");
    TORCH_CHECK(bias_tensor.dim() == 1, "Bias must have shape (K,).");
    TORCH_CHECK(bias_tensor.size(0) == output.size(1), "Bias output channels must match output.");
    return output + bias_tensor.view({1, -1, 1, 1});
}

torch::Tensor dietconv_v1_forward_impl(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding
) {
    validate_inputs(input, weight);
    TORCH_CHECK(stride > 0, "stride must be positive.");
    TORCH_CHECK(padding >= 0, "padding must be non-negative.");

    const auto [pad_h, pad_w] = to_pair(padding);
    auto x = input.contiguous();
    auto w = weight.contiguous();
    auto padded = make_padded_input(x, pad_h, pad_w);

    const int64_t batch_size = x.size(0);
    const int64_t channels = x.size(1);
    const int64_t kernel_height = w.size(2);
    const int64_t kernel_width = w.size(3);
    const int64_t out_channels = w.size(0);
    const int64_t padded_height = padded.size(2);
    const int64_t padded_width = padded.size(3);
    const int64_t out_h = (padded_height - kernel_height) / stride + 1;
    const int64_t out_w = (padded_width - kernel_width) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());
    auto kernel_slices =
        w.permute({3, 0, 1, 2}).contiguous().view({kernel_width, out_channels, channels * kernel_height});

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        auto padded_batch = padded[batch];
        auto output_batch = output[batch];
        for (int64_t out_y = 0; out_y < out_h; ++out_y) {
            const int64_t row_start = out_y * stride;
            auto temp = padded_batch.narrow(1, row_start, kernel_height).contiguous().view({channels * kernel_height, padded_width});
            auto row_out = torch::zeros({out_channels, out_w}, x.options());
            for (int64_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
                torch::Tensor strip;
                if (stride == 1) {
                    strip = temp.narrow(1, kernel_x, out_w);
                } else {
                    strip = temp.index({Slice(), Slice(kernel_x, kernel_x + stride * out_w, stride)}).contiguous();
                }
                auto partial = at::mm(kernel_slices[kernel_x], strip);
                if (kernel_x == 0) {
                    row_out.copy_(partial);
                } else {
                    row_out.add_(partial);
                }
            }
            output_batch.select(1, out_y).copy_(row_out);
        }
    }
    return add_bias_if_present(output, bias);
}

torch::Tensor dietconv_v2_forward_impl(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t tile_out_width
) {
    validate_inputs(input, weight);
    TORCH_CHECK(stride > 0, "stride must be positive.");
    TORCH_CHECK(padding >= 0, "padding must be non-negative.");

    const auto [pad_h, pad_w] = to_pair(padding);
    auto x = input.contiguous();
    auto w = weight.contiguous();
    auto padded = make_padded_input(x, pad_h, pad_w);

    const int64_t batch_size = x.size(0);
    const int64_t channels = x.size(1);
    const int64_t kernel_height = w.size(2);
    const int64_t kernel_width = w.size(3);
    const int64_t out_channels = w.size(0);
    const int64_t padded_height = padded.size(2);
    const int64_t padded_width = padded.size(3);
    const int64_t out_h = (padded_height - kernel_height) / stride + 1;
    const int64_t out_w = (padded_width - kernel_width) / stride + 1;
    const int64_t resolved_tile_out_width = resolve_tile_out_width(out_w, stride, tile_out_width);

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());
    auto kernel_slices =
        w.permute({3, 0, 1, 2}).contiguous().view({kernel_width, out_channels, channels * kernel_height});

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        auto padded_batch = padded[batch];
        auto output_batch = output[batch];
        for (int64_t out_y = 0; out_y < out_h; ++out_y) {
            const int64_t row_start = out_y * stride;
            for (int64_t tile_start = 0; tile_start < out_w; tile_start += resolved_tile_out_width) {
                const int64_t tile_w = std::min(resolved_tile_out_width, out_w - tile_start);
                const int64_t input_x = tile_start * stride;
                const int64_t tile_input_w = (tile_w - 1) * stride + kernel_width;
                auto temp = padded_batch
                                .narrow(1, row_start, kernel_height)
                                .narrow(2, input_x, tile_input_w)
                                .contiguous()
                                .view({channels * kernel_height, tile_input_w});
                auto tile_out = torch::zeros({out_channels, tile_w}, x.options());
                for (int64_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
                    torch::Tensor strip;
                    if (stride == 1) {
                        strip = temp.narrow(1, kernel_x, tile_w);
                    } else {
                        strip = temp.index({Slice(), Slice(kernel_x, kernel_x + stride * tile_w, stride)}).contiguous();
                    }
                    auto partial = at::mm(kernel_slices[kernel_x], strip);
                    if (kernel_x == 0) {
                        tile_out.copy_(partial);
                    } else {
                        tile_out.add_(partial);
                    }
                }
                output_batch.select(1, out_y).narrow(1, tile_start, tile_w).copy_(tile_out);
            }
        }
    }
    return add_bias_if_present(output, bias);
}

}  // namespace

torch::Tensor dietconv_v1_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding
) {
    return dietconv_v1_forward_impl(input, weight, bias, stride, padding);
}

torch::Tensor dietconv_v2_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t tile_out_width
) {
    return dietconv_v2_forward_impl(input, weight, bias, stride, padding, tile_out_width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dietconv_v1_forward", &dietconv_v1_forward, "DietConv v1 forward (CPU)");
    m.def("dietconv_v2_forward", &dietconv_v2_forward, "DietConv v2 forward (CPU)");
}
