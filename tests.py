#!/usr/bin/env python3

from models import EncoderF64_Res, EncoderF32_Res, EncoderF_Res, EncoderZ_Res, Colorizer
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
    def test_EncoderZ_Res():
        x = torch.randn(4, 1, 256, 256)
        model = EncoderZ_Res()
        model.float()
        y = model(x)
        assert y.shape == torch.Size([4, 119])

    @commenter
    def test_EncoderF16():
        x = torch.randn(4, 1, 256, 256)
        model = EncoderF_Res()
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
