#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

struct Problem {
    int c = 0;
    int h = 0;
    int w = 0;
    int k = 0;
    int fh = 0;
    int fw = 0;
    int stride = 1;
    int pad = 0;

    int padded_h() const { return h + 2 * pad; }
    int padded_w() const { return w + 2 * pad; }
    int out_h() const { return (padded_h() - fh) / stride + 1; }
    int out_w() const { return (padded_w() - fw) / stride + 1; }
    int kernel_inner() const { return c * fh * fw; }
    int slice_inner() const { return c * fh; }
};

struct PreparedData {
    Problem problem;
    std::vector<float> padded_input;
    std::vector<float> weight_flat;
    std::vector<float> kernel_slices;
};

struct RunResult {
    std::vector<float> output;
    std::size_t workspace_bytes = 0;
    int tile_out_width = 0;
};

int resolve_tile_out_width(const Problem& problem, int tile_out_width) {
    if (tile_out_width > 0) {
        return std::min(tile_out_width, problem.out_w());
    }
    const int heuristic = problem.stride == 1 ? 64 : 32;
    return std::min(heuristic, problem.out_w());
}

std::size_t padded_index(const Problem& problem, int channel, int y, int x) {
    return (static_cast<std::size_t>(channel) * problem.padded_h() + y) * problem.padded_w() + x;
}

std::size_t input_index(const Problem& problem, int channel, int y, int x) {
    return (static_cast<std::size_t>(channel) * problem.h + y) * problem.w + x;
}

std::size_t weight_index(const Problem& problem, int out_channel, int in_channel, int fy, int fx) {
    return (((static_cast<std::size_t>(out_channel) * problem.c + in_channel) * problem.fh + fy) * problem.fw) + fx;
}

std::size_t output_index(const Problem& problem, int out_channel, int out_y, int out_x) {
    return (static_cast<std::size_t>(out_channel) * problem.out_h() + out_y) * problem.out_w() + out_x;
}

template <typename Worker>
void parallel_for_rows(int threads, int rows, Worker worker) {
    if (threads <= 1 || rows <= 1) {
        worker(0, 0, rows);
        return;
    }

    const int actual_threads = std::min(threads, rows);
    std::vector<std::thread> pool;
    pool.reserve(actual_threads);
    const int base_rows = rows / actual_threads;
    const int remainder = rows % actual_threads;
    int start = 0;
    for (int thread_id = 0; thread_id < actual_threads; ++thread_id) {
        const int count = base_rows + (thread_id < remainder ? 1 : 0);
        const int end = start + count;
        pool.emplace_back([=, &worker]() { worker(thread_id, start, end); });
        start = end;
    }
    for (auto& thread : pool) {
        thread.join();
    }
}

PreparedData prepare_problem(const Problem& problem, int seed) {
    PreparedData prepared;
    prepared.problem = problem;

    std::mt19937 rng(seed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    std::vector<float> input(static_cast<std::size_t>(problem.c) * problem.h * problem.w);
    prepared.weight_flat.resize(static_cast<std::size_t>(problem.k) * problem.kernel_inner());
    for (auto& value : input) {
        value = distribution(rng);
    }
    for (auto& value : prepared.weight_flat) {
        value = distribution(rng);
    }

    prepared.padded_input.assign(static_cast<std::size_t>(problem.c) * problem.padded_h() * problem.padded_w(), 0.0f);
    for (int channel = 0; channel < problem.c; ++channel) {
        for (int y = 0; y < problem.h; ++y) {
            for (int x = 0; x < problem.w; ++x) {
                prepared.padded_input[padded_index(problem, channel, y + problem.pad, x + problem.pad)] =
                    input[input_index(problem, channel, y, x)];
            }
        }
    }

    prepared.kernel_slices.resize(static_cast<std::size_t>(problem.fw) * problem.k * problem.slice_inner());
    for (int fx = 0; fx < problem.fw; ++fx) {
        for (int out_channel = 0; out_channel < problem.k; ++out_channel) {
            for (int in_channel = 0; in_channel < problem.c; ++in_channel) {
                for (int fy = 0; fy < problem.fh; ++fy) {
                    const std::size_t slice_index =
                        ((static_cast<std::size_t>(fx) * problem.k + out_channel) * problem.slice_inner()) +
                        (in_channel * problem.fh + fy);
                    prepared.kernel_slices[slice_index] =
                        prepared.weight_flat[weight_index(problem, out_channel, in_channel, fy, fx)];
                }
            }
        }
    }

    return prepared;
}

RunResult run_im2col(const PreparedData& prepared, int threads) {
    const Problem& problem = prepared.problem;
    const int out_h = problem.out_h();
    const int out_w = problem.out_w();
    const int out_hw = out_h * out_w;
    const int inner = problem.kernel_inner();

    std::vector<float> cols(static_cast<std::size_t>(inner) * out_hw);
    parallel_for_rows(threads, out_h, [&](int, int start_row, int end_row) {
        for (int out_y = start_row; out_y < end_row; ++out_y) {
            for (int out_x = 0; out_x < out_w; ++out_x) {
                const int col_index = out_y * out_w + out_x;
                const int in_y = out_y * problem.stride;
                const int in_x = out_x * problem.stride;
                for (int channel = 0; channel < problem.c; ++channel) {
                    for (int fy = 0; fy < problem.fh; ++fy) {
                        for (int fx = 0; fx < problem.fw; ++fx) {
                            const int row_index = ((channel * problem.fh + fy) * problem.fw) + fx;
                            cols[static_cast<std::size_t>(row_index) * out_hw + col_index] =
                                prepared.padded_input[padded_index(problem, channel, in_y + fy, in_x + fx)];
                        }
                    }
                }
            }
        }
    });

    std::vector<float> matrix_out(static_cast<std::size_t>(problem.k) * out_hw, 0.0f);
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        problem.k,
        out_hw,
        inner,
        1.0f,
        prepared.weight_flat.data(),
        inner,
        cols.data(),
        out_hw,
        0.0f,
        matrix_out.data(),
        out_hw
    );

    std::vector<float> output(static_cast<std::size_t>(problem.k) * out_hw, 0.0f);
    for (int out_channel = 0; out_channel < problem.k; ++out_channel) {
        for (int out_y = 0; out_y < out_h; ++out_y) {
            for (int out_x = 0; out_x < out_w; ++out_x) {
                const int col_index = out_y * out_w + out_x;
                output[output_index(problem, out_channel, out_y, out_x)] =
                    matrix_out[static_cast<std::size_t>(out_channel) * out_hw + col_index];
            }
        }
    }

    return {std::move(output), cols.size() * sizeof(float), 0};
}

RunResult run_dietconv_v1(const PreparedData& prepared, int threads) {
    const Problem& problem = prepared.problem;
    const int out_h = problem.out_h();
    const int out_w = problem.out_w();
    const int slice_inner = problem.slice_inner();
    const int padded_w = problem.padded_w();

    std::vector<float> output(static_cast<std::size_t>(problem.k) * out_h * out_w, 0.0f);
    parallel_for_rows(threads, out_h, [&](int, int start_row, int end_row) {
        std::vector<float> temp(static_cast<std::size_t>(slice_inner) * padded_w, 0.0f);
        std::vector<float> strip(static_cast<std::size_t>(slice_inner) * out_w, 0.0f);
        std::vector<float> row_out(static_cast<std::size_t>(problem.k) * out_w, 0.0f);

        for (int out_y = start_row; out_y < end_row; ++out_y) {
            const int in_y = out_y * problem.stride;
            for (int row = 0; row < slice_inner; ++row) {
                const int channel = row / problem.fh;
                const int fy = row % problem.fh;
                const float* src = &prepared.padded_input[padded_index(problem, channel, in_y + fy, 0)];
                std::copy(src, src + padded_w, temp.begin() + static_cast<std::size_t>(row) * padded_w);
            }

            std::fill(row_out.begin(), row_out.end(), 0.0f);
            for (int fx = 0; fx < problem.fw; ++fx) {
                for (int row = 0; row < slice_inner; ++row) {
                    const float* temp_row = temp.data() + static_cast<std::size_t>(row) * padded_w + fx;
                    float* strip_row = strip.data() + static_cast<std::size_t>(row) * out_w;
                    for (int out_x = 0; out_x < out_w; ++out_x) {
                        strip_row[out_x] = temp_row[out_x * problem.stride];
                    }
                }
                cblas_sgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    problem.k,
                    out_w,
                    slice_inner,
                    1.0f,
                    prepared.kernel_slices.data() + (static_cast<std::size_t>(fx) * problem.k * slice_inner),
                    slice_inner,
                    strip.data(),
                    out_w,
                    fx == 0 ? 0.0f : 1.0f,
                    row_out.data(),
                    out_w
                );
            }

            for (int out_channel = 0; out_channel < problem.k; ++out_channel) {
                float* dst = &output[output_index(problem, out_channel, out_y, 0)];
                const float* src = row_out.data() + static_cast<std::size_t>(out_channel) * out_w;
                std::copy(src, src + out_w, dst);
            }
        }
    });

    const std::size_t per_thread_bytes =
        static_cast<std::size_t>(slice_inner) * (padded_w + out_w) * sizeof(float);
    return {std::move(output), static_cast<std::size_t>(threads) * per_thread_bytes, problem.out_w()};
}

RunResult run_dietconv_v2(const PreparedData& prepared, int threads, int tile_out_width) {
    const Problem& problem = prepared.problem;
    const int out_h = problem.out_h();
    const int out_w = problem.out_w();
    const int slice_inner = problem.slice_inner();
    const int resolved_tile_out_width = resolve_tile_out_width(problem, tile_out_width);
    const int max_tile_w = resolved_tile_out_width;
    const int max_tile_input_w = (max_tile_w - 1) * problem.stride + problem.fw;
    const bool can_slide_rows = problem.stride > 0 && problem.stride < problem.fh;
    const bool can_use_direct_temp_gemm = problem.stride == 1;

    std::vector<float> output(static_cast<std::size_t>(problem.k) * out_h * out_w, 0.0f);
    parallel_for_rows(threads, out_h, [&](int, int start_row, int end_row) {
        std::vector<float> temp(static_cast<std::size_t>(slice_inner) * max_tile_input_w, 0.0f);
        std::vector<float> strip(static_cast<std::size_t>(slice_inner) * max_tile_w, 0.0f);
        std::vector<float> tile_out(static_cast<std::size_t>(problem.k) * max_tile_w, 0.0f);

        auto fill_temp_window = [&](int base_in_y, int input_x, int tile_input_w) {
            for (int row = 0; row < slice_inner; ++row) {
                const int channel = row / problem.fh;
                const int fy = row % problem.fh;
                const float* src = &prepared.padded_input[padded_index(problem, channel, base_in_y + fy, input_x)];
                std::copy(src, src + tile_input_w, temp.begin() + static_cast<std::size_t>(row) * max_tile_input_w);
            }
        };

        auto slide_temp_window = [&](int next_base_in_y, int input_x, int tile_input_w) {
            if (!can_slide_rows) {
                fill_temp_window(next_base_in_y, input_x, tile_input_w);
                return;
            }
            const int preserved_rows = problem.fh - problem.stride;
            for (int channel = 0; channel < problem.c; ++channel) {
                float* channel_base = temp.data() + static_cast<std::size_t>(channel) * problem.fh * max_tile_input_w;
                std::memmove(
                    channel_base,
                    channel_base + static_cast<std::size_t>(problem.stride) * max_tile_input_w,
                    static_cast<std::size_t>(preserved_rows) * max_tile_input_w * sizeof(float)
                );
                for (int fy = preserved_rows; fy < problem.fh; ++fy) {
                    const int src_y = next_base_in_y + fy;
                    float* dst = channel_base + static_cast<std::size_t>(fy) * max_tile_input_w;
                    const float* src = &prepared.padded_input[padded_index(problem, channel, src_y, input_x)];
                    std::copy(src, src + tile_input_w, dst);
                }
            }
        };

        for (int tile_start = 0; tile_start < out_w; tile_start += resolved_tile_out_width) {
            const int tile_w = std::min(resolved_tile_out_width, out_w - tile_start);
            const int input_x = tile_start * problem.stride;
            const int tile_input_w = (tile_w - 1) * problem.stride + problem.fw;

            bool temp_initialized = false;
            for (int out_y = start_row; out_y < end_row; ++out_y) {
                const int in_y = out_y * problem.stride;
                if (!temp_initialized) {
                    fill_temp_window(in_y, input_x, tile_input_w);
                    temp_initialized = true;
                } else {
                    slide_temp_window(in_y, input_x, tile_input_w);
                }

                std::fill(tile_out.begin(), tile_out.end(), 0.0f);
                for (int fx = 0; fx < problem.fw; ++fx) {
                    const float* lowering_matrix = nullptr;
                    int lowering_leading_dim = max_tile_w;
                    if (can_use_direct_temp_gemm) {
                        lowering_matrix = temp.data() + fx;
                        lowering_leading_dim = max_tile_input_w;
                    } else {
                        for (int row = 0; row < slice_inner; ++row) {
                            const float* temp_row = temp.data() + static_cast<std::size_t>(row) * max_tile_input_w + fx;
                            float* strip_row = strip.data() + static_cast<std::size_t>(row) * max_tile_w;
                            for (int out_x = 0; out_x < tile_w; ++out_x) {
                                strip_row[out_x] = temp_row[out_x * problem.stride];
                            }
                        }
                        lowering_matrix = strip.data();
                    }

                    cblas_sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        problem.k,
                        tile_w,
                        slice_inner,
                        1.0f,
                        prepared.kernel_slices.data() + (static_cast<std::size_t>(fx) * problem.k * slice_inner),
                        slice_inner,
                        lowering_matrix,
                        lowering_leading_dim,
                        fx == 0 ? 0.0f : 1.0f,
                        tile_out.data(),
                        max_tile_w
                    );
                }

                for (int out_channel = 0; out_channel < problem.k; ++out_channel) {
                    float* dst = &output[output_index(problem, out_channel, out_y, tile_start)];
                    const float* src = tile_out.data() + static_cast<std::size_t>(out_channel) * max_tile_w;
                    std::copy(src, src + tile_w, dst);
                }
            }
        }
    });

    const std::size_t per_thread_bytes = can_use_direct_temp_gemm
        ? static_cast<std::size_t>(slice_inner) * max_tile_input_w * sizeof(float)
        : static_cast<std::size_t>(slice_inner) * (max_tile_input_w + max_tile_w) * sizeof(float);
    return {std::move(output), static_cast<std::size_t>(threads) * per_thread_bytes, resolved_tile_out_width};
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double diff = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        diff = std::max(diff, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    return diff;
}

double checksum(const std::vector<float>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0);
}

std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> args;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key.rfind("--", 0) != 0 || index + 1 >= argc) {
            throw std::runtime_error("Arguments must be passed as --key value pairs.");
        }
        args[key.substr(2)] = argv[++index];
    }
    return args;
}

int get_int_arg(const std::unordered_map<std::string, std::string>& args, const std::string& key, int fallback) {
    auto found = args.find(key);
    if (found == args.end()) {
        return fallback;
    }
    return std::stoi(found->second);
}

std::string get_string_arg(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    const std::string& fallback
) {
    auto found = args.find(key);
    if (found == args.end()) {
        return fallback;
    }
    return found->second;
}

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        const std::string backend = get_string_arg(args, "backend", "im2col");
        const int repeat = get_int_arg(args, "repeat", 3);
        const int warmup = get_int_arg(args, "warmup", 1);
        const int threads = get_int_arg(args, "threads", 1);
        const int seed = get_int_arg(args, "seed", 0);
        const int tile_out_width = get_int_arg(args, "tile-out-width", 32);

        setenv("VECLIB_MAXIMUM_THREADS", "1", 0);

        Problem problem;
        problem.c = get_int_arg(args, "c", 3);
        problem.h = get_int_arg(args, "h", 64);
        problem.w = get_int_arg(args, "w", 64);
        problem.k = get_int_arg(args, "k", 64);
        problem.fh = get_int_arg(args, "fh", 3);
        problem.fw = get_int_arg(args, "fw", 3);
        problem.stride = get_int_arg(args, "stride", 1);
        problem.pad = get_int_arg(args, "pad", 0);

        if (threads < 1) {
            throw std::runtime_error("threads must be >= 1");
        }
        if (tile_out_width < 0) {
            throw std::runtime_error("tile-out-width must be >= 0");
        }

        const PreparedData prepared = prepare_problem(problem, seed);
        const RunResult baseline = run_im2col(prepared, 1);

        std::function<RunResult()> runner;
        if (backend == "im2col") {
            runner = [&]() { return run_im2col(prepared, threads); };
        } else if (backend == "dietconv-v1") {
            runner = [&]() { return run_dietconv_v1(prepared, threads); };
        } else if (backend == "dietconv-v2") {
            runner = [&]() { return run_dietconv_v2(prepared, threads, tile_out_width); };
        } else {
            throw std::runtime_error("Unknown backend: " + backend);
        }

        for (int iteration = 0; iteration < warmup; ++iteration) {
            runner();
        }

        std::vector<double> timings;
        timings.reserve(repeat);
        RunResult measured;
        for (int iteration = 0; iteration < repeat; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            measured = runner();
            const auto end = std::chrono::steady_clock::now();
            timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        const double mean = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
        double variance = 0.0;
        for (double timing : timings) {
            const double delta = timing - mean;
            variance += delta * delta;
        }
        variance /= timings.size();
        const double stddev = std::sqrt(variance);
        const double diff = max_abs_diff(measured.output, baseline.output);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "{"
                  << "\"backend\":\"" << backend << "\","
                  << "\"threads\":" << threads << ","
                  << "\"tile_out_width\":" << measured.tile_out_width << ","
                  << "\"c\":" << problem.c << ","
                  << "\"h\":" << problem.h << ","
                  << "\"w\":" << problem.w << ","
                  << "\"k\":" << problem.k << ","
                  << "\"fh\":" << problem.fh << ","
                  << "\"fw\":" << problem.fw << ","
                  << "\"stride\":" << problem.stride << ","
                  << "\"pad\":" << problem.pad << ","
                  << "\"out_h\":" << problem.out_h() << ","
                  << "\"out_w\":" << problem.out_w() << ","
                  << "\"mean_ms\":" << mean << ","
                  << "\"std_ms\":" << stddev << ","
                  << "\"workspace_bytes\":" << measured.workspace_bytes << ","
                  << "\"workspace_mib\":" << (static_cast<double>(measured.workspace_bytes) / (1024.0 * 1024.0)) << ","
                  << "\"max_abs_diff_vs_im2col\":" << diff << ","
                  << "\"checksum\":" << checksum(measured.output)
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
