import os 
import pickle
import models
import argparse
import random

import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='m_test5')
    parser.add_argument('--detail', default='multi gpu')

    # Mode
    parser.add_argument('--norm_type', default='instance', 
            choices=['instance', 'batch', 'layer'])

    # IO
    parser.add_argument('--path_log', default='runs_test')
    parser.add_argument('--path_ckpts', default='ckpts')
    parser.add_argument('--path_config', default='config.pickle')
    parser.add_argument('--path_ckpt_g', default='./pretrained/G_ema_256.pth')
    parser.add_argument('--path_ckpt_d', default='./pretrained/D_256.pth')
    parser.add_argument('--path_imgnet_train', default='./imgnet/train')
    parser.add_argument('--path_imgnet_val', default='./imgnet/val')

    parser.add_argument('--index_target',
            # type=int, nargs='+', default=[11,14,15])
            type=int, nargs='+', default=[15])
    parser.add_argument('--num_worker', default=8)
    parser.add_argument('--iter_sample', default=4)

    # Others
    parser.add_argument('--dim_z', type=int, default=119)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size_batch', default=32)
    parser.add_argument('--device', default='cuda:0')

    return parser.parse_args()

def main(args):
    with open(args.path_config, 'rb') as f:
        config = pickle.load(f)

    dev = args.device

    G = models.Generator(**config)
    G.load_state_dict(torch.load(args.path_ckpt_g), strict=False)
    G.to(dev)
    G.eval()

    z = torch.zeros((args.size_batch, args.dim_z)).to(dev)
    z.normal_(mean=0, std=0.5)

    c = [random.randint(14, 15) for _ in range(args.size_batch)]
    c = [ 15 for _ in range(args.size_batch)]
    c = torch.LongTensor(c).to(dev)
    c = G.shared(c)

    output = G.forward(z, c)
    print(c.shape)
    output = (output - output.min()) / output.max()
    grid = make_grid(output, nrow=8, normalize=True)
    im = ToPILImage()(grid)
    im.save("sample.jpg")

if __name__ == '__main__':
    args = parse_args()
    main(args)
