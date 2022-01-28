#!/usr/bin/env python3

from models import EncoderF32_Res, EncoderF_Res, EncoderZ_Res
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
