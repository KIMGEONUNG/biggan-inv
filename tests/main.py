import unittest 
from unittest import TestCase
from utils import to_gray
from torchvision.transforms import Grayscale
import torch


class UtilTest(TestCase):

    def test_trivial(self):

        x = torch.randn(1, 4, 4)
        print(x)
        print(x.repeat(3,1,1))




    def test_to_gray(self):
        dim_batch = 8 
        dim_spatial = 16
        to_gray_gt = Grayscale()

        xs = [torch.randn(3, dim_spatial, dim_spatial),
              torch.randn(dim_batch, 3, dim_spatial, dim_spatial)]

        for x in xs:
            x_g_gt = to_gray_gt(x)
            x_g_hat = to_gray(x)

            a = torch.isclose(x_g_gt, x_g_hat)
            a = torch.all(a).item()

            self.assertTrue(a)


if __name__ == '__main__':
    unittest.main()
