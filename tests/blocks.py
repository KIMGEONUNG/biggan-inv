import unittest
from unittest import TestCase
from models import ResConvBlock
import torch
from hashlib import sha256
from global_config import PATH_STORAGE, LEN_HASH
from os.path import join


class Tester(TestCase):

    def setup(self):
        pass

    def test_ResBlock_1(self):
        # bc432d47ed
        id_code = sha256('v001'.encode('utf-8')).hexdigest()[:LEN_HASH]
        
        x = torch.load(join(PATH_STORAGE, '%s_input_0') % id_code)
        output = torch.load(join(PATH_STORAGE, '%s_output_0') % id_code)

        model = ResConvBlock(6, 10, is_down=True, use_res=True)
        model.eval()
        name_model = type(model).__name__
        model.load_state_dict(
                torch.load(join(PATH_STORAGE, '%s_m_ResConvBlock') % (id_code, name_model)),
                strict=True)

        y = model(x)
        self.assertTrue(torch.equal(output, y))

    def test_ResBlock_2(self):
        # 947d22951c
        id_code = sha256('v002'.encode('utf-8')).hexdigest()[:LEN_HASH]

        x = torch.load(join(PATH_STORAGE, '%s_input_0') % id_code)
        output = torch.load(join(PATH_STORAGE, '%s_output_0') % id_code)

        model = ResConvBlock(6, 10, is_down=True, use_res=False)
        model.eval()
        name_model = type(model).__name__
        model.load_state_dict(
                torch.load(join(PATH_STORAGE, '%s_m_ResConvBlock') % (id_code, name_model)),
                strict=True)

        y = model(x)
        self.assertTrue(torch.equal(output, y))


if __name__ == '__main__':
    unittest.main()
