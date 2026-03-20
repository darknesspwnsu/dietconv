import unittest

import numpy as np

from dietconv.algorithms import conv2d_dietconv, conv2d_direct, conv2d_im2col


class ConvAlgorithmTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(1234)
        self.x = rng.standard_normal((3, 10, 9), dtype=np.float32)
        self.weight = rng.standard_normal((4, 3, 3, 3), dtype=np.float32)

    def test_im2col_matches_direct(self) -> None:
        direct = conv2d_direct(self.x, self.weight, stride=1, padding=1)
        actual = conv2d_im2col(self.x, self.weight, stride=1, padding=1)
        np.testing.assert_allclose(actual, direct, rtol=1e-5, atol=1e-5)

    def test_dietconv_matches_direct(self) -> None:
        direct = conv2d_direct(self.x, self.weight, stride=1, padding=1)
        actual = conv2d_dietconv(self.x, self.weight, stride=1, padding=1)
        np.testing.assert_allclose(actual, direct, rtol=1e-5, atol=1e-5)

    def test_dietconv_matches_im2col_with_stride(self) -> None:
        actual = conv2d_dietconv(self.x, self.weight, stride=2, padding=1)
        expected = conv2d_im2col(self.x, self.weight, stride=2, padding=1)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
