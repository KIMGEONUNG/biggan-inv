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
    parser.add_argument('--std', type=int, default=0.9)
    parser.add_argument('--dim_z', type=int, default=119)
    parser.add_argument('--dim_z_tail', type=int, default=17)
    parser.add_argument('--index_min', type=int, default=17*3)
    parser.add_argument('--index_max', type=int, default=17*7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size_batch', default=8)
    parser.add_argument('--device', default='cuda:1')

    return parser.parse_args()


def set_seed(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main(args):

    if args.seed >= 0:
        set_seed(args.seed)


    with open(args.path_config, 'rb') as f:
        config = pickle.load(f)

    dev = args.device

    G = models.Generator(**config)
    G.load_state_dict(torch.load(args.path_ckpt_g), strict=False)
    G.to(dev)
    G.eval()


    c = [random.randint(0, 1000) for _ in range(args.size_batch)]
    c = torch.LongTensor(c).to(dev)
    c = G.shared(c)

    z = torch.zeros((args.size_batch, args.dim_z)).to(dev)
    z.normal_(mean=0, std=args.std)

    outputs = []
    for i in range(8):
        z_ = torch.zeros((args.size_batch, args.dim_z)).to(dev)
        z_.normal_(mean=0, std=args.std)
        z[:, args.index_min:args.index_max] = z_[:, args.index_min:args.index_max]

        output = G.forward(z, c)
        output = (output - output.min()) / output.max()
        outputs.append(output.detach().cpu())

    output = torch.cat(outputs, dim=0) 
    grid = make_grid(output, nrow=args.size_batch, normalize=True)
    im = ToPILImage()(grid)
    im.save("z_impact/sample_%d_%d.jpg" % (args.index_min, args.index_max))


if __name__ == '__main__':
    args = parse_args()
    main(args)
