import unittest
from unittest import TestCase

from models import ResConvBlock

import torch


class Tester(TestCase):

    def setup(self):
        pass

    def test_ResBlock_1(self):
        x = torch.randn(4, 6, 256, 256)
        model = ResConvBlock(6, 10, is_down=True)
        y = model(x)
        assert y.shape == torch.Size([4, 10, 128, 128])

    def test_ResBlock_2(self):
        x = torch.randn(4, 6, 256, 256)
        model = ResConvBlock(6, 10, is_down=True, use_res=False)
        y = model(x)
        assert y.shape == torch.Size([4, 10, 128, 128])

    def test_ResBlock_3(self):
        id_code = 'bc432d47ed'

        x = torch.load('./tests/storage/%s_input_0' % (id_code))
        output = torch.load('./tests/storage/%s_output_0' % (id_code))

        model = ResConvBlock(6, 10, is_down=True, use_res=True)
        model.eval()
        model.load_state_dict(
                torch.load('./tests/storage/%s_m_ResConvBlock' % (id_code)),
                strict=True)

        y = model(x)
        self.assertTrue(torch.equal(output, y))

    def test_ResBlock_4(self):
        id_code = '947d22951c'

        x = torch.load('./tests/storage/%s_input_0' % (id_code))
        output = torch.load('./tests/storage/%s_output_0' % (id_code))

        model = ResConvBlock(6, 10, is_down=True, use_res=False)
        model.eval()
        model.load_state_dict(
                torch.load('./tests/storage/%s_m_ResConvBlock' % (id_code)),
                strict=True)

        y = model(x)
        self.assertTrue(torch.equal(output, y))



if __name__ == '__main__':
    unittest.main()
