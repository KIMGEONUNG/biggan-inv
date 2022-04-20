#!/usr/bin/env python3

import unittest
from unittest import TestCase
from models import EncoderF_Res
                   
import torch
from typing import Optional


class UtilsTest(TestCase):

    def test_transpose(self):

        # shape[3, 2, 2]
        map_initial = torch.LongTensor(
                        [
                            [[0, 1], [2, 3]],
                            [[3, 2], [1, 0]],
                            [[1, 0], [3, 2]]
                        ]
                    )
        print(map_initial.shape)
        input = torch.LongTensor(
                    [
                        [[1, 0, 0, 0], [0, 1, 0, 0]],
                        [[0, 0, 1, 0], [0, 0, 0, 1]]
                    ])
        print(input.shape)


class CustomTests(TestCase):

    def setUp(self):
        pass

    def test_tmp(self):
        print(EncoderF64_Res)


if __name__ == '__main__':
    unittest.main()
