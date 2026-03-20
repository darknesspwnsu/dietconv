import unittest

import torch
import torch.nn.functional as F

from dietconv.torch_ops import (
    DietConv2dV1Compiled,
    DietConv2dV2Compiled,
    DietConv2dV2,
    autotune_dietconv_v2_tile_width,
    can_use_dietconv_system_optimizations,
    clear_dietconv_autotune_cache,
    prepack_dietconv_weight,
    dietconv2d_v1_compiled,
    dietconv2d_v1_compiled_prepacked,
    dietconv2d_v2,
    dietconv2d_v2_compiled,
    dietconv2d_v2_compiled_autotuned_prepacked,
    dietconv2d_v2_compiled_prepacked,
    load_dietconv_extension,
    unfold_conv2d,
)


class TorchOpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_dietconv_extension()

    def setUp(self) -> None:
        torch.manual_seed(4321)
        clear_dietconv_autotune_cache()
        self.x = torch.randn(1, 3, 10, 9)
        self.weight = torch.randn(4, 3, 3, 3)

    def test_unfold_matches_conv2d(self) -> None:
        expected = F.conv2d(self.x, self.weight, stride=1, padding=1)
        actual = unfold_conv2d(self.x, self.weight, stride=1, padding=1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_dietconv_v2_matches_conv2d(self) -> None:
        expected = F.conv2d(self.x, self.weight, stride=1, padding=1)
        actual = dietconv2d_v2(self.x, self.weight, stride=1, padding=1, tile_out_width=4)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_dietconv_v2_matches_conv2d_with_stride(self) -> None:
        expected = F.conv2d(self.x, self.weight, stride=2, padding=1)
        actual = dietconv2d_v2(self.x, self.weight, stride=2, padding=1, tile_out_width=3)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_module_wrapper_matches_function(self) -> None:
        module = DietConv2dV2(3, 4, 3, stride=1, padding=1, bias=False, tile_out_width=4)
        with torch.no_grad():
            module.weight.copy_(self.weight)
        actual = module(self.x)
        expected = dietconv2d_v2(self.x, self.weight, stride=1, padding=1, tile_out_width=4)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_compiled_dietconv_v1_matches_conv2d(self) -> None:
        expected = F.conv2d(self.x, self.weight, stride=1, padding=1)
        actual = dietconv2d_v1_compiled(self.x, self.weight, stride=1, padding=1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_compiled_dietconv_v2_matches_conv2d(self) -> None:
        expected = F.conv2d(self.x, self.weight, stride=1, padding=1)
        actual = dietconv2d_v2_compiled(self.x, self.weight, stride=1, padding=1, tile_out_width=4)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_compiled_variants_match_stride_case(self) -> None:
        x = torch.randn(2, 3, 11, 11)
        weight = torch.randn(5, 3, 3, 3)
        expected = F.conv2d(x, weight, stride=2, padding=1)
        actual_v1 = dietconv2d_v1_compiled(x, weight, stride=2, padding=1)
        actual_v2 = dietconv2d_v2_compiled(x, weight, stride=2, padding=1, tile_out_width=3)
        torch.testing.assert_close(actual_v1, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual_v2, expected, rtol=1e-5, atol=1e-5)

    def test_compiled_v2_matches_python_v2(self) -> None:
        expected = dietconv2d_v2(self.x, self.weight, stride=1, padding=1, tile_out_width=4)
        actual = dietconv2d_v2_compiled(self.x, self.weight, stride=1, padding=1, tile_out_width=4)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_compiled_modules_match_functions(self) -> None:
        module_v1 = DietConv2dV1Compiled(3, 4, 3, stride=1, padding=1, bias=False)
        module_v2 = DietConv2dV2Compiled(3, 4, 3, stride=1, padding=1, bias=False, tile_out_width=4)
        with torch.no_grad():
            module_v1.weight.copy_(self.weight)
            module_v2.weight.copy_(self.weight)
        actual_v1 = module_v1(self.x)
        actual_v2 = module_v2(self.x)
        expected_v1 = dietconv2d_v1_compiled(self.x, self.weight, stride=1, padding=1)
        expected_v2 = dietconv2d_v2_compiled(self.x, self.weight, stride=1, padding=1, tile_out_width=4)
        torch.testing.assert_close(actual_v1, expected_v1, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual_v2, expected_v2, rtol=1e-5, atol=1e-5)

    def test_prepacked_compiled_paths_match_wrappers(self) -> None:
        packed = prepack_dietconv_weight(self.weight)
        actual_v1 = dietconv2d_v1_compiled_prepacked(self.x, packed, stride=1, padding=1)
        actual_v2 = dietconv2d_v2_compiled_prepacked(self.x, packed, stride=1, padding=1, tile_out_width=4)
        expected_v1 = dietconv2d_v1_compiled(self.x, self.weight, stride=1, padding=1)
        expected_v2 = dietconv2d_v2_compiled(self.x, self.weight, stride=1, padding=1, tile_out_width=4)
        torch.testing.assert_close(actual_v1, expected_v1, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual_v2, expected_v2, rtol=1e-5, atol=1e-5)

    def test_optimization_gate_matches_supported_case(self) -> None:
        packed = prepack_dietconv_weight(self.weight)
        self.assertTrue(
            can_use_dietconv_system_optimizations(self.x, packed, stride=1, padding=1)
        )
        self.assertFalse(
            can_use_dietconv_system_optimizations(self.x.double(), packed, stride=1, padding=1)
        )

    def test_autotuned_prepacked_path_matches_explicit_compiled_path(self) -> None:
        packed = prepack_dietconv_weight(self.weight)
        tile = autotune_dietconv_v2_tile_width(self.x, packed, stride=1, padding=1)
        actual = dietconv2d_v2_compiled_autotuned_prepacked(self.x, packed, stride=1, padding=1)
        expected = dietconv2d_v2_compiled_prepacked(
            self.x,
            packed,
            stride=1,
            padding=1,
            tile_out_width=tile,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_compiled_v2_module_autotunes_once_per_shape(self) -> None:
        module = DietConv2dV2Compiled(3, 4, 3, stride=1, padding=1, bias=False, tile_out_width=0)
        with torch.no_grad():
            module.weight.copy_(self.weight)
        _ = module(self.x)
        first_cache = dict(module._autotuned_tile_widths)
        _ = module(self.x)
        self.assertEqual(first_cache, module._autotuned_tile_widths)
        self.assertEqual(len(module._autotuned_tile_widths), 1)


if __name__ == "__main__":
    unittest.main()
