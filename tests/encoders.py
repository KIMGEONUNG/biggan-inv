import unittest
from unittest import TestCase
from models import EncoderF_Res
import torch
from hashlib import sha256
from global_config import PATH_STORAGE, LEN_HASH
from os.path import join


class Tester(TestCase):

    def setup(self):
        pass

    def test_Encoder_1(self):
        id_code = sha256('v003'.encode('utf-8')).hexdigest()[:LEN_HASH]
        
        x = torch.load(join(PATH_STORAGE, '%s_input_0') % id_code)
        output = torch.load(join(PATH_STORAGE, '%s_output_0') % id_code)

        model = EncoderF_Res()
        model.eval()
        name_model = type(model).__name__
        model.load_state_dict(
                torch.load(join(PATH_STORAGE, '%s_m_%s') % (id_code, name_model)),
                strict=True)

        y = model(x)
        self.assertTrue(torch.equal(output, y))

    def test_Encoder_2(self):
        id_code = sha256('v004'.encode('utf-8')).hexdigest()[:LEN_HASH]
        
        x = torch.load(join(PATH_STORAGE, '%s_input_0') % id_code)
        c = torch.load(join(PATH_STORAGE, '%s_input_1') % id_code)
        output = torch.load(join(PATH_STORAGE, '%s_output_0') % id_code)

        model = EncoderF_Res(norm='adabatch')
        model.eval()
        name_model = type(model).__name__
        model.load_state_dict(
                torch.load(join(PATH_STORAGE, '%s_m_%s') % (id_code, name_model)),
                strict=True)

        y = model(x, c)
        self.assertTrue(torch.equal(output, y))

    def test_Encoder_3(self):
        id_code = sha256('v005'.encode('utf-8')).hexdigest()[:LEN_HASH]
        
        x = torch.load(join(PATH_STORAGE, '%s_input_0') % id_code)
        c = torch.load(join(PATH_STORAGE, '%s_input_1') % id_code)
        output = torch.load(join(PATH_STORAGE, '%s_output_0') % id_code)

        model = EncoderF_Res(norm='adabatch', ch_c=77)
        model.eval()
        name_model = type(model).__name__
        model.load_state_dict(
                torch.load(join(PATH_STORAGE, '%s_m_%s') % (id_code, name_model)),
                strict=True)

        y = model(x, c)
        self.assertTrue(torch.equal(output, y))

    def test_Encoder_4(self):
        id_code = sha256('v007'.encode('utf-8')).hexdigest()[:LEN_HASH]
        
        x = torch.load(join(PATH_STORAGE, '%s_input_0') % id_code)
        c = torch.load(join(PATH_STORAGE, '%s_input_1') % id_code)
        z = torch.load(join(PATH_STORAGE, '%s_input_2') % id_code)
        output = torch.load(join(PATH_STORAGE, '%s_output_0') % id_code)

        model = EncoderF_Res(norm='adabatch', ch_c=60, z_chunk_size=10)
        model.eval()
        name_model = type(model).__name__
        model.load_state_dict(
                torch.load(join(PATH_STORAGE, '%s_m_%s') % (id_code, name_model)),
                strict=True)

        y = model(x, c, z)
        self.assertTrue(torch.equal(output, y))


if __name__ == '__main__':
    unittest.main()
