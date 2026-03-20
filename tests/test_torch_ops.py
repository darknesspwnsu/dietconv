import unittest

import torch
import torch.nn.functional as F

from dietconv.torch_ops import DietConv2dV2, dietconv2d_v2, unfold_conv2d


class TorchOpTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4321)
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


if __name__ == "__main__":
    unittest.main()
