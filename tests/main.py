#!/usr/bin/env python3

import unittest
from unittest import TestCase
from models import EncoderF64_Res
                   
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


from models import (EncoderF64_Res, EncoderF32_Res, EncoderF, EncoderZ_Res,
                    Colorizer, ConvBlock)
import torch
import re


def commenter(fn_test):

    def wrapper(self):
        print('Started Test : %s' % fn_test.__name__)
        fn_test()
        print('Finished Test : %s' % fn_test.__name__)

    return wrapper


class Tester():

    @commenter
    def test_ResBlock_1():
        x = torch.randn(4, 6, 256, 256)
        model = ConvBlock(6, 10, is_down=True)
        model.float()
        y = model(x)
        assert y.shape == torch.Size([4, 10, 128, 128])

    @commenter
    def test_ResBlock_2():
        x = torch.randn(4, 6, 256, 256)
        model = ConvBlock(6, 10, is_down=True, use_res=False)
        model.float()
        y = model(x)
        assert y.shape == torch.Size([4, 10, 128, 128])

    @commenter
    def test_EncoderZ_Res():
        x = torch.randn(4, 1, 256, 256)
        model = EncoderZ_Res()
        model.float()
        y = model(x)
        assert y.shape == torch.Size([4, 119])

    @commenter
    def test_EncoderF16():
        x = torch.randn(4, 1, 256, 256)
        model = EncoderF()
        model.float()
        y = model(x)
        assert y.shape == torch.Size([4, 768, 16, 16])

    @commenter
    def test_EncoderF32():
        x = torch.randn(4, 1, 256, 256)
        model = EncoderF32_Res()
        model.float()
        y = model(x)
        assert y.shape == torch.Size([4, 768, 32, 32])

    @commenter
    def test_EncoderF64():
        x = torch.randn(4, 1, 256, 256)
        model = EncoderF64_Res()
        model.float()
        y = model(x)
        assert y.shape == torch.Size([4, 384, 64, 64])

    @commenter
    def test_Colorizer_1():
        import pickle
        with open('./pretrained/config.pickle', 'rb') as f:
            config = pickle.load(f)
        EG = Colorizer(config,
                      './pretrained/G_ema_256.pth',
                      'adabatch',
                      id_mid_layer=2,
                      activation='relu',
                      fix_g=False,
                      init_e=None,
                      use_attention=False,
                      dim_f=1)
        EG.float()
        c = torch.LongTensor([1, 2, 3, 4])
        z = torch.zeros((4, 119))
        z.normal_(mean=0, std=0.8)
        x = torch.randn(4, 1, 256, 256) 
        y = EG(x, c, z)
        assert y.shape == torch.Size([4, 3, 256, 256])

    @commenter
    def test_Colorizer_8():
        import pickle
        with open('./pretrained/config.pickle', 'rb') as f:
            config = pickle.load(f)
        EG = Colorizer(config,
                      './pretrained/G_ema_256.pth',
                      'adabatch',
                      id_mid_layer=2,
                      activation='relu',
                      fix_g=False,
                      init_e=None,
                      use_attention=False,
                      dim_f=8)
        EG.float()
        c = torch.LongTensor([1, 2, 3, 4])
        z = torch.zeros((4, 119))
        z.normal_(mean=0, std=0.8)
        x = torch.randn(4, 1, 256, 256) 
        y = EG(x, c, z)
        assert y.shape == torch.Size([4, 3, 256, 256])

    @commenter
    def test_Colorizer_16():
        import pickle
        with open('./pretrained/config.pickle', 'rb') as f:
            config = pickle.load(f)
        EG = Colorizer(config,
                      './pretrained/G_ema_256.pth',
                      'adabatch',
                      id_mid_layer=2,
                      activation='relu',
                      fix_g=False,
                      init_e=None,
                      use_attention=False,
                      use_res=False,
                      use_cond_e=False,
                      dim_f=16)
        EG.float()
        c = torch.LongTensor([1, 2, 3, 4])
        z = torch.zeros((4, 119))
        z.normal_(mean=0, std=0.8)
        x = torch.randn(4, 1, 256, 256) 
        y = EG(x, c, z)
        assert y.shape == torch.Size([4, 3, 256, 256])

    @commenter
    def test_Colorizer_16():
        import pickle
        with open('./pretrained/config.pickle', 'rb') as f:
            config = pickle.load(f)
        EG = Colorizer(config,
                      './pretrained/G_ema_256.pth',
                      'adabatch',
                      id_mid_layer=2,
                      activation='relu',
                      fix_g=False,
                      init_e=None,
                      use_attention=False,
                      dim_f=16)
        EG.float()
        c = torch.LongTensor([1, 2, 3, 4])
        z = torch.zeros((4, 119))
        z.normal_(mean=0, std=0.8)
        x = torch.randn(4, 1, 256, 256) 
        y = EG(x, c, z)
        assert y.shape == torch.Size([4, 3, 256, 256])

    @commenter
    def test_Colorizer_32():
        import pickle
        with open('./pretrained/config.pickle', 'rb') as f:
            config = pickle.load(f)
        EG = Colorizer(config,
                      './pretrained/G_ema_256.pth',
                      'adabatch',
                      id_mid_layer=2,
                      activation='relu',
                      fix_g=False,
                      init_e=None,
                      use_attention=False,
                      dim_f=32)
        EG.float()
        c = torch.LongTensor([1, 2, 3, 4])
        z = torch.zeros((4, 119))
        z.normal_(mean=0, std=0.8)
        x = torch.randn(4, 1, 256, 256) 
        y = EG(x, c, z)
        assert y.shape == torch.Size([4, 3, 256, 256])

    @commenter
    def test_Colorizer_64():
        import pickle
        with open('./pretrained/config.pickle', 'rb') as f:
            config = pickle.load(f)
        EG = Colorizer(config,
                      './pretrained/G_ema_256.pth',
                      'adabatch',
                      id_mid_layer=2,
                      activation='relu',
                      fix_g=False,
                      init_e=None,
                      use_attention=False,
                      dim_f=64)
        EG.float()
        c = torch.LongTensor([1, 2, 3, 4])
        z = torch.zeros((4, 119))
        z.normal_(mean=0, std=0.8)
        x = torch.randn(4, 1, 256, 256) 
        y = EG(x, c, z)
        assert y.shape == torch.Size([4, 3, 256, 256])


def main():
    tester = Tester()
    fns = [getattr(tester, n) for n in dir(tester) if re.match('^test', n)]
    for fn in fns:
        try:
            fn()
        except Exception as e:
            print('\tTest Fail', e)


if __name__ == '__main__':
    main()

