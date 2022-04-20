import unittest
from unittest import TestCase
from models import Colorizer
import torch
from hashlib import sha256
from global_config import PATH_STORAGE, LEN_HASH
from os.path import join
import pickle


class Tester(TestCase):

    def setUp(self):
        with open('./pretrained/config.pickle', 'rb') as f:
            self.config = pickle.load(f)
        self.path_ckpt_G = './pretrained/G_ema_256.pth'

    def test_model_1(self):
        id_code = sha256('v006'.encode('utf-8')).hexdigest()[:LEN_HASH]

        x = torch.load(join(PATH_STORAGE, '%s_input_0') % id_code)
        c = torch.load(join(PATH_STORAGE, '%s_input_1') % id_code)
        z = torch.load(join(PATH_STORAGE, '%s_input_2') % id_code)
        output = torch.load(join(PATH_STORAGE, '%s_output_0') % id_code)

        model = Colorizer(self.config, self.path_ckpt_G)
        model.eval()
        name_model = type(model).__name__
        model.load_state_dict(
                torch.load(join(PATH_STORAGE, '%s_m_%s') % (id_code, name_model)),
                strict=True)

        y = model(x, c, z)
        self.assertTrue(torch.equal(output, y))

    def test_model_2(self):
        id_code = sha256('v008'.encode('utf-8')).hexdigest()[:LEN_HASH]

        x = torch.load(join(PATH_STORAGE, '%s_input_0') % id_code)
        c = torch.load(join(PATH_STORAGE, '%s_input_1') % id_code)
        z_g = torch.load(join(PATH_STORAGE, '%s_input_2') % id_code)
        z_e = torch.load(join(PATH_STORAGE, '%s_input_3') % id_code)
        output = torch.load(join(PATH_STORAGE, '%s_output_0') % id_code)

        model = Colorizer(self.config, self.path_ckpt_G, dim_encoder_c=145, chunk_size_z_e=17)
        model.eval()
        name_model = type(model).__name__
        model.load_state_dict(
                torch.load(join(PATH_STORAGE, '%s_m_%s') % (id_code, name_model)),
                strict=True)

        y = model(x, c, z_g, z_e)
        self.assertTrue(torch.equal(output, y))


if __name__ == '__main__':
    unittest.main()
