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
        # bc432d47ed
        id_code = sha256('v003'.encode('utf-8')).hexdigest()[:LEN_HASH]

        join(PATH_STORAGE, '%s_input_0') % id_code
        
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

if __name__ == '__main__':
    unittest.main()
