from __future__ import annotations

import os
import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


def _pair(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def _resolve_tile_out_width(out_w: int, stride: int, tile_out_width: int) -> int:
    if tile_out_width > 0:
        return min(tile_out_width, out_w)
    heuristic = 64 if stride == 1 else 32
    return min(heuristic, out_w)


_EXTENSION = None
_V2_TILE_CACHE: dict[tuple[int, ...], int] = {}


def load_dietconv_extension():
    global _EXTENSION
    if _EXTENSION is None:
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        build_directory = os.path.join(root, "build", "torch_extension")
        os.makedirs(build_directory, exist_ok=True)
        _EXTENSION = load(
            name="dietconv_torch_ext",
            sources=[os.path.join(root, "cpp", "torch_dietconv_extension.cpp")],
            build_directory=build_directory,
            extra_cflags=["-O3", "-std=c++17", "-Wno-deprecated-declarations"],
            extra_ldflags=["-framework", "Accelerate"],
            verbose=False,
        )
    return _EXTENSION


def prepack_dietconv_weight(weight: torch.Tensor) -> torch.Tensor:
    extension = load_dietconv_extension()
    return extension.prepack_weight(weight)


def can_use_dietconv_system_optimizations(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
) -> bool:
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    if x.device.type != "cpu" or packed_weight.device.type != "cpu":
        return False
    if x.dtype != torch.float32 or packed_weight.dtype != torch.float32:
        return False
    if x.dim() != 4 or packed_weight.dim() != 3:
        return False
    if stride_h != stride_w or pad_h != pad_w:
        return False
    if stride_h <= 0 or pad_h < 0:
        return False
    if x.numel() == 0 or packed_weight.numel() == 0:
        return False
    if packed_weight.shape[2] % x.shape[1] != 0:
        return False
    kernel_height = packed_weight.shape[2] // x.shape[1]
    kernel_width = packed_weight.shape[0]
    padded_h = x.shape[-2] + 2 * pad_h
    padded_w = x.shape[-1] + 2 * pad_w
    if padded_h < kernel_height or padded_w < kernel_width:
        return False
    return True


def _v2_cache_key(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    stride: int,
    padding: int,
) -> tuple[int, ...]:
    return (
        int(x.shape[0]),
        int(x.shape[1]),
        int(x.shape[2]),
        int(x.shape[3]),
        int(packed_weight.shape[0]),
        int(packed_weight.shape[1]),
        int(packed_weight.shape[2]),
        int(stride),
        int(padding),
        int(torch.get_num_threads()),
    )


def candidate_tile_out_widths(out_w: int, stride: int) -> list[int]:
    base = [16, 24, 32, 48, 64, 96, 128, out_w]
    if stride > 1:
        base = [8, 16, 24, 32, 48, out_w]
    return sorted({min(width, out_w) for width in base if width > 0})


def clear_dietconv_autotune_cache() -> None:
    _V2_TILE_CACHE.clear()


def autotune_dietconv_v2_tile_width(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
    repeat: int = 1,
    warmup: int = 1,
    reuse_cache: bool = True,
) -> int:
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    if stride_h != stride_w:
        raise ValueError("DietConv v2 autotune requires equal height/width stride.")
    if pad_h != pad_w:
        raise ValueError("DietConv v2 autotune requires equal height/width padding.")

    out_w = (x.shape[-1] + 2 * pad_w - packed_weight.shape[0]) // stride_w + 1
    if not can_use_dietconv_system_optimizations(x, packed_weight, stride=stride_w, padding=pad_w):
        return _resolve_tile_out_width(out_w, stride_w, 0)
    key = _v2_cache_key(x, packed_weight, stride_w, pad_w)
    if reuse_cache and key in _V2_TILE_CACHE:
        return _V2_TILE_CACHE[key]

    trials: list[tuple[int, float, int]] = []
    with torch.no_grad():
        for width in candidate_tile_out_widths(out_w, stride_w):
            for _ in range(warmup):
                dietconv2d_v2_compiled_prepacked(
                    x,
                    packed_weight,
                    stride=stride_w,
                    padding=pad_w,
                    tile_out_width=width,
                )
            timings = []
            for _ in range(repeat):
                start = time.perf_counter()
                dietconv2d_v2_compiled_prepacked(
                    x,
                    packed_weight,
                    stride=stride_w,
                    padding=pad_w,
                    tile_out_width=width,
                )
                timings.append((time.perf_counter() - start) * 1000.0)
            mean_ms = sum(timings) / len(timings)
            workspace_bytes = workspace_bytes_dietconv2d_v2(
                x,
                packed_weight,
                stride=stride_w,
                padding=pad_w,
                tile_out_width=width,
            )
            trials.append((width, mean_ms, workspace_bytes))

    fastest_ms = min(mean_ms for _, mean_ms, _ in trials)
    viable = [trial for trial in trials if trial[1] <= fastest_ms * 1.05]
    chosen = min(viable, key=lambda trial: (trial[2], trial[0]))[0]
    if reuse_cache:
        _V2_TILE_CACHE[key] = chosen
    return chosen


def workspace_bytes_dietconv2d_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
    tile_out_width: int = 0,
) -> int:
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    if stride_h != stride_w:
        raise ValueError("This torch DietConv demo requires equal height/width stride.")
    out_w = (x.shape[-1] + 2 * pad_w - weight.shape[-1]) // stride_w + 1
    resolved_tile_w = _resolve_tile_out_width(out_w, stride_w, tile_out_width)
    tile_input_w = (resolved_tile_w - 1) * stride_w + weight.shape[-1]
    return int(x.element_size() * x.shape[-3] * weight.shape[-2] * tile_input_w)


def workspace_bytes_dietconv2d_v1(
    x: torch.Tensor,
    weight: torch.Tensor,
    padding: int | Tuple[int, int] = 0,
) -> int:
    pad_h, pad_w = _pair(padding)
    padded_width = x.shape[-1] + 2 * pad_w
    return int(x.element_size() * x.shape[-3] * weight.shape[-2] * padded_width)


def workspace_bytes_unfold(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
) -> int:
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    out_h = (x.shape[-2] + 2 * pad_h - weight.shape[-2]) // stride_h + 1
    out_w = (x.shape[-1] + 2 * pad_w - weight.shape[-1]) // stride_w + 1
    return int(
        x.element_size()
        * x.shape[0]
        * x.shape[1]
        * weight.shape[-2]
        * weight.shape[-1]
        * out_h
        * out_w
    )


def unfold_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
) -> torch.Tensor:
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    n, c, h, w = x.shape
    out_channels, weight_channels, kernel_height, kernel_width = weight.shape
    if c != weight_channels:
        raise ValueError("Input channel count and weight channel count must match.")

    cols = F.unfold(
        x,
        kernel_size=(kernel_height, kernel_width),
        padding=(pad_h, pad_w),
        stride=(stride_h, stride_w),
    )
    flat_weight = weight.view(out_channels, -1)
    out = torch.einsum("oc,bcn->bon", flat_weight, cols)
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    out_h = (h + 2 * pad_h - kernel_height) // stride_h + 1
    out_w = (w + 2 * pad_w - kernel_width) // stride_w + 1
    return out.view(n, out_channels, out_h, out_w)


def dietconv2d_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
    tile_out_width: int = 0,
) -> torch.Tensor:
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    if stride_h != stride_w:
        raise ValueError("This torch DietConv demo requires equal height/width stride.")
    if x.dim() != 4:
        raise ValueError("Expected input with shape (N, C, H, W).")

    n, channels, height, width = x.shape
    out_channels, weight_channels, kernel_height, kernel_width = weight.shape
    if channels != weight_channels:
        raise ValueError("Input channel count and weight channel count must match.")

    padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h))
    out_h = (height + 2 * pad_h - kernel_height) // stride_h + 1
    out_w = (width + 2 * pad_w - kernel_width) // stride_w + 1
    resolved_tile_out_width = _resolve_tile_out_width(out_w, stride_w, tile_out_width)

    output = x.new_zeros((n, out_channels, out_h, out_w))
    kernel_slices = weight.permute(3, 0, 1, 2).contiguous().view(
        kernel_width,
        out_channels,
        channels * kernel_height,
    )

    for batch in range(n):
        for out_y in range(out_h):
            row_start = out_y * stride_h
            for tile_start in range(0, out_w, resolved_tile_out_width):
                tile_w = min(resolved_tile_out_width, out_w - tile_start)
                input_x = tile_start * stride_w
                tile_input_w = (tile_w - 1) * stride_w + kernel_width
                temp = padded[
                    batch,
                    :,
                    row_start : row_start + kernel_height,
                    input_x : input_x + tile_input_w,
                ].contiguous().view(channels * kernel_height, tile_input_w)
                tile_out = x.new_zeros((out_channels, tile_w))
                for kernel_x in range(kernel_width):
                    if stride_w == 1:
                        strip = temp[:, kernel_x : kernel_x + tile_w]
                    else:
                        strip = temp[:, kernel_x : kernel_x + stride_w * tile_w : stride_w].contiguous()
                    tile_out = tile_out + kernel_slices[kernel_x] @ strip
                output[batch, :, out_y, tile_start : tile_start + tile_w] = tile_out

    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    return output


def dietconv2d_v1_compiled(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
) -> torch.Tensor:
    packed_weight = prepack_dietconv_weight(weight)
    return dietconv2d_v1_compiled_prepacked(
        x,
        packed_weight,
        bias=bias,
        stride=stride,
        padding=padding,
    )


def dietconv2d_v1_compiled_prepacked(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
) -> torch.Tensor:
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    if stride_h != stride_w:
        raise ValueError("Compiled DietConv v1 requires equal height/width stride.")
    if pad_h != pad_w:
        raise ValueError("Compiled DietConv v1 requires equal height/width padding.")
    extension = load_dietconv_extension()
    return extension.dietconv_v1_prepacked_forward(x, packed_weight, bias, stride_h, pad_h)


def dietconv2d_v2_compiled(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
    tile_out_width: int = 0,
) -> torch.Tensor:
    packed_weight = prepack_dietconv_weight(weight)
    return dietconv2d_v2_compiled_prepacked(
        x,
        packed_weight,
        bias=bias,
        stride=stride,
        padding=padding,
        tile_out_width=tile_out_width,
    )


def dietconv2d_v2_compiled_prepacked(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
    tile_out_width: int = 0,
) -> torch.Tensor:
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    if stride_h != stride_w:
        raise ValueError("Compiled DietConv v2 requires equal height/width stride.")
    if pad_h != pad_w:
        raise ValueError("Compiled DietConv v2 requires equal height/width padding.")
    extension = load_dietconv_extension()
    return extension.dietconv_v2_prepacked_forward(x, packed_weight, bias, stride_h, pad_h, tile_out_width)


def dietconv2d_v2_compiled_autotuned(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
    repeat: int = 1,
    warmup: int = 1,
    reuse_cache: bool = True,
) -> torch.Tensor:
    packed_weight = prepack_dietconv_weight(weight)
    return dietconv2d_v2_compiled_autotuned_prepacked(
        x,
        packed_weight,
        bias=bias,
        stride=stride,
        padding=padding,
        repeat=repeat,
        warmup=warmup,
        reuse_cache=reuse_cache,
    )


def dietconv2d_v2_compiled_autotuned_prepacked(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
    repeat: int = 1,
    warmup: int = 1,
    reuse_cache: bool = True,
) -> torch.Tensor:
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    if stride_h != stride_w:
        raise ValueError("Compiled DietConv v2 autotune requires equal height/width stride.")
    if pad_h != pad_w:
        raise ValueError("Compiled DietConv v2 autotune requires equal height/width padding.")
    tile_out_width = autotune_dietconv_v2_tile_width(
        x,
        packed_weight,
        stride=stride_h,
        padding=pad_h,
        repeat=repeat,
        warmup=warmup,
        reuse_cache=reuse_cache,
    )
    return dietconv2d_v2_compiled_prepacked(
        x,
        packed_weight,
        bias=bias,
        stride=stride_h,
        padding=pad_h,
        tile_out_width=tile_out_width,
    )


class DietConv2dV2(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        tile_out_width: int = 0,
    ) -> None:
        super().__init__()
        kernel_height, kernel_width = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.tile_out_width = tile_out_width
        self.weight = torch.nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_height, kernel_width)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_channels * kernel_height * kernel_width
            bound = 1 / fan_in**0.5
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dietconv2d_v2(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            tile_out_width=self.tile_out_width,
        )


class DietConv2dV1Compiled(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        kernel_height, kernel_width = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight = torch.nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_height, kernel_width)
        )
        self.register_buffer("_packed_weight", torch.empty(0), persistent=False)
        self._packed_weight_version = -1
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_channels * kernel_height * kernel_width
            bound = 1 / fan_in**0.5
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def _refresh_packed_weight(self) -> None:
        if self._packed_weight_version != self.weight._version:
            self._packed_weight = prepack_dietconv_weight(self.weight)
            self._packed_weight_version = self.weight._version

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._refresh_packed_weight()
        return dietconv2d_v1_compiled_prepacked(
            x,
            self._packed_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )


class DietConv2dV2Compiled(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        tile_out_width: int = 0,
    ) -> None:
        super().__init__()
        kernel_height, kernel_width = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.tile_out_width = tile_out_width
        self.weight = torch.nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_height, kernel_width)
        )
        self.register_buffer("_packed_weight", torch.empty(0), persistent=False)
        self._packed_weight_version = -1
        self._autotuned_tile_widths: dict[tuple[int, ...], int] = {}
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_channels * kernel_height * kernel_width
            bound = 1 / fan_in**0.5
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def _refresh_packed_weight(self) -> None:
        if self._packed_weight_version != self.weight._version:
            self._packed_weight = prepack_dietconv_weight(self.weight)
            self._packed_weight_version = self.weight._version
            self._autotuned_tile_widths.clear()

    def _resolve_runtime_tile_width(self, x: torch.Tensor) -> int:
        if self.tile_out_width > 0:
            return self.tile_out_width
        if not can_use_dietconv_system_optimizations(
            x,
            self._packed_weight,
            stride=self.stride,
            padding=self.padding,
        ):
            out_w = (x.shape[-1] + 2 * self.padding - self.weight.shape[-1]) // self.stride + 1
            return _resolve_tile_out_width(out_w, self.stride, 0)
        cache_key = (
            int(x.shape[0]),
            int(x.shape[1]),
            int(x.shape[2]),
            int(x.shape[3]),
            int(torch.get_num_threads()),
        )
        if cache_key not in self._autotuned_tile_widths:
            self._autotuned_tile_widths[cache_key] = autotune_dietconv_v2_tile_width(
                x,
                self._packed_weight,
                stride=self.stride,
                padding=self.padding,
            )
        return self._autotuned_tile_widths[cache_key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._refresh_packed_weight()
        tile_out_width = self._resolve_runtime_tile_width(x)
        return dietconv2d_v2_compiled_prepacked(
            x,
            self._packed_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            tile_out_width=tile_out_width,
        )
