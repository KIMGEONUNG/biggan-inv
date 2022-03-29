import unittest
from unittest import TestCase
from utils import (SemanticDataset, extract,
                    label_to_one_hot_label)
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
        # print(input.shape)
        # result = torch.transpose(input, 0, 2)
        # print(result.shape)
        # answer = torch.IntTensor(
        #             [
        #                 [[1, 0, 0, 0], [0, 1, 0, 0]],
        #                 [[0, 0, 1, 0], [0, 0, 0, 1]]
        #             ])
        # exit()
        #
        # self.assertEqual(result, answer)


class CustomTests(TestCase):

    def setUp(self):
        pass

    def test_tmp(self):
        dataset = SemanticDataset('dataset/train', 'train', 'semantic_map_train')
        dataset = extract(dataset, list(range(10, 20)))
        iterator = iter(dataset)
        a = next(iterator)


if __name__ == '__main__':
    unittest.main()
