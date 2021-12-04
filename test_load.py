import os
from skimage import color
import numpy as np
from torch.utils.data import DataLoader
import models
from encoders import EncoderF_16
import torch
import pickle
import argparse
from train import Colorizer


def parse():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument('--path_config', default='config.pickle')
    parser.add_argument('--path_ckpt_eg', default='./ckpts/EG_0011400.ckpt')

    parser.add_argument('--num_layer', default=2)
    parser.add_argument('--norm_type', default='instance', 
            choices=['instance', 'batch', 'layer'])

    return parser.parse_args()


def main(args):
    print(args)

    with open(args.path_config, 'rb') as f:
        config = pickle.load(f)

    EG = Colorizer(config, args.path_ckpt_eg, args.norm_type,
            id_mid_layer=args.num_layer)

    EG.load_state_dict(torch.load(args.path_ckpt_eg), strict=True)
    # Load Configuratuion


if __name__ == '__main__':
    args = parse()
    main(args)
