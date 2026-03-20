from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _pair(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


@dataclass(frozen=True)
class Conv2DProblem:
    name: str
    input_channels: int
    input_height: int
    input_width: int
    output_channels: int
    kernel_height: int
    kernel_width: int
    stride: int = 1
    padding: int = 0

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (self.input_channels, self.input_height, self.input_width)

    @property
    def weight_shape(self) -> Tuple[int, int, int, int]:
        return (
            self.output_channels,
            self.input_channels,
            self.kernel_height,
            self.kernel_width,
        )

    @property
    def padded_spatial_shape(self) -> Tuple[int, int]:
        pad_h, pad_w = _pair(self.padding)
        return (self.input_height + 2 * pad_h, self.input_width + 2 * pad_w)

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        pad_h, pad_w = _pair(self.padding)
        out_h = (self.input_height + 2 * pad_h - self.kernel_height) // self.stride + 1
        out_w = (self.input_width + 2 * pad_w - self.kernel_width) // self.stride + 1
        return (self.output_channels, out_h, out_w)


def pad_input(x: np.ndarray, padding: int | Tuple[int, int]) -> np.ndarray:
    pad_h, pad_w = _pair(padding)
    if pad_h == 0 and pad_w == 0:
        return x
    return np.pad(
        x,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
    )


def workspace_bytes_im2col(
    x: np.ndarray,
    weight: np.ndarray,
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
) -> int:
    padded = pad_input(x, padding)
    _, height, width = padded.shape
    _, channels, kernel_height, kernel_width = weight.shape
    out_h = (height - kernel_height) // stride + 1
    out_w = (width - kernel_width) // stride + 1
    return int(x.dtype.itemsize * channels * kernel_height * kernel_width * out_h * out_w)


def workspace_bytes_dietconv(
    x: np.ndarray,
    weight: np.ndarray,
    padding: int | Tuple[int, int] = 0,
) -> int:
    padded = pad_input(x, padding)
    channels, _, width = padded.shape
    kernel_height = weight.shape[2]
    return int(x.dtype.itemsize * channels * kernel_height * width)


def im2col_matrix(
    x: np.ndarray,
    kernel_size: Tuple[int, int],
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
) -> np.ndarray:
    kernel_height, kernel_width = kernel_size
    padded = pad_input(x, padding)
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (kernel_height, kernel_width),
        axis=(1, 2),
    )
    windows = windows[:, ::stride, ::stride, :, :]
    return np.ascontiguousarray(
        windows.transpose(0, 3, 4, 1, 2).reshape(
            padded.shape[0] * kernel_height * kernel_width,
            -1,
        )
    )


def conv2d_im2col(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
) -> np.ndarray:
    cols = im2col_matrix(x, weight.shape[2:], stride=stride, padding=padding)
    flat_weight = weight.reshape(weight.shape[0], -1)
    out = flat_weight @ cols
    if bias is not None:
        out += bias[:, None]
    padded = pad_input(x, padding)
    out_h = (padded.shape[1] - weight.shape[2]) // stride + 1
    out_w = (padded.shape[2] - weight.shape[3]) // stride + 1
    return out.reshape(weight.shape[0], out_h, out_w)


def conv2d_dietconv(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
) -> np.ndarray:
    padded = pad_input(x, padding)
    channels, _, padded_width = padded.shape
    out_channels, weight_channels, kernel_height, kernel_width = weight.shape
    if channels != weight_channels:
        raise ValueError("Input channel count and weight channel count must match.")

    out_h = (padded.shape[1] - kernel_height) // stride + 1
    out_w = (padded_width - kernel_width) // stride + 1
    output = np.zeros((out_channels, out_h, out_w), dtype=x.dtype)

    # Reorder the filter bank so each kernel-width slice can be consumed by GEMM.
    kernel_slices = np.ascontiguousarray(weight.transpose(3, 0, 1, 2)).reshape(
        kernel_width,
        out_channels,
        channels * kernel_height,
    )

    for out_y in range(out_h):
        row_start = out_y * stride
        temp = np.ascontiguousarray(
            padded[:, row_start : row_start + kernel_height, :]
        )
        for kernel_x in range(kernel_width):
            strip = temp[:, :, kernel_x : kernel_x + stride * out_w : stride]
            strip_matrix = np.ascontiguousarray(strip).reshape(
                channels * kernel_height,
                out_w,
            )
            output[:, out_y, :] += kernel_slices[kernel_x] @ strip_matrix

    if bias is not None:
        output += bias[:, None, None]
    return output


def conv2d_direct(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
) -> np.ndarray:
    padded = pad_input(x, padding)
    out_channels, weight_channels, kernel_height, kernel_width = weight.shape
    channels, _, _ = padded.shape
    if channels != weight_channels:
        raise ValueError("Input channel count and weight channel count must match.")

    out_h = (padded.shape[1] - kernel_height) // stride + 1
    out_w = (padded.shape[2] - kernel_width) // stride + 1
    output = np.zeros((out_channels, out_h, out_w), dtype=x.dtype)

    for out_channel in range(out_channels):
        for out_y in range(out_h):
            row_start = out_y * stride
            for out_x in range(out_w):
                col_start = out_x * stride
                patch = padded[
                    :,
                    row_start : row_start + kernel_height,
                    col_start : col_start + kernel_width,
                ]
                output[out_channel, out_y, out_x] = np.sum(
                    patch * weight[out_channel]
                )
        if bias is not None:
            output[out_channel] += bias[out_channel]
    return output
