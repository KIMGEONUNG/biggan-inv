import models
from encoders import EncoderF_16

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader 

import pickle
import argparse
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchvision.transforms import (ToPILImage, Compose, ToTensor,
        Resize, CenterCrop)
from tqdm import tqdm

"""
# Dimension infos
    z: ([8, 17])
    h: ([8, 24576])
    index 0 : ([8, 1536, 4, 4])
    index 1 : ([8, 1536, 8, 8])
    index 2 : ([8, 768, 16, 16])
    index 3 : ([8, 768, 32, 32])
    index 4 : ([8, 384, 64, 64])
    index 5 : ([8, 192, 128, 128])
    result: ([8, 96, 256, 256])
"""
LAYER_DIM = {
        0: [1536, 4],
        1: [1536, 8],
        2: [768, 16],
        3: [768, 32],
        4: [384, 64],
        4: [192, 128],
        }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='encoder_f_16')

    # Mode
    parser.add_argument('--mode', default='wip', 
            choices=['sampling', 'critic', 'wip', 'train'])

    # IO
    parser.add_argument('--path_config', default='config.pickle')
    parser.add_argument('--path_ckpt_g', default='./pretrained/G_ema_256.pth')
    parser.add_argument('--path_ckpt_d', default='./pretrained/D_256.pth')
    parser.add_argument('--path_imgnet_train', default='./imgnet/train')
    parser.add_argument('--path_imgnet_val', default='./imgnet/val')
    
    # Encoder Traning
    parser.add_argument('--path_dataset_encoder', default='./dataset_encoder/')
    parser.add_argument('--path_dataset_encoder_val', default='./dataset_encoder_val/')
    parser.add_argument('--class_index', default=15)
    parser.add_argument('--num_layer', default=2)
    parser.add_argument('--num_epoch', default=400)
    parser.add_argument('--interval_save', default=3)

    # Optimizer
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--lr_d", type=float, default=0.0001)
    parser.add_argument("--b1_d", type=float, default=0.5)
    parser.add_argument("--b2_d", type=float, default=0.999)

    # Verbose
    parser.add_argument('--print_config', default=False)
    parser.add_argument('--print_generator', default=True)
    parser.add_argument('--print_discriminator', default=True)

    # loader
    parser.add_argument('--use_pretrained_d', default=True)
    parser.add_argument('--use_pretrained_g', default=True)

    # Loss
    parser.add_argument('--loss_mse', action='store_true', default=True)
    parser.add_argument('--loss_lpips', action='store_true', default=True)
    parser.add_argument('--loss_adv', action='store_true', default=True)

    # Loss coef
    parser.add_argument('--coef_mse', type=float, default=1.0)
    parser.add_argument('--coef_lpips', type=float, default=0.05)
    parser.add_argument('--coef_gen', type=float, default=0.1)
    parser.add_argument('--coef_hsv', type=float, default=1.0)
    parser.add_argument('--coef_adv', type=float, default=1.0)

    # Others
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size_batch', default=32)
    parser.add_argument('--w_class', default=True)
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


def critic(G, D, config, args, dev):
    """
    Discriminator using loss, not score
    """

    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (20,3)

    num_iter = 32
    num_bin = 40

    # Make Eval
    G.eval().to(dev)
    D.eval().to(dev)

    # Real image critics  
    transform = Compose([
        Resize(256),
        CenterCrop(256),
        ToTensor(),
        ])

    plt.subplot(1, 2, 1)
    dataset = ImageFolder(args.path_imgnet_train, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.size_batch, shuffle=True)
    critics = []
    for i in range(num_iter):
        x_real, y = next(iter(dataloader))
        x_real, y = x_real.to(dev), y.to(dev).long()
        x_real = (x_real - 0.5) * 2

        if not args.w_class:
            y = None
        with torch.no_grad():
            critic_val, _ = D(x_real, y)
        critics.append(critic_val.detach().cpu())

    critics = torch.cat(critics, dim=0).view(-1).numpy()
    plt.hist(critics, num_bin, alpha = 0.5)

    # Fake critics 
    critics = []
    # plt.subplot(1, 2, 2)
    for i in range(num_iter):
        # Sample latent
        z = torch.zeros((args.size_batch, G.dim_z)).to(dev)
        std = config['sample_std']  # 0.5
        z.normal_(mean=0, std=std)

        n_class = config['n_classes']  # 1000
        y = torch.zeros(args.size_batch).to(dev).long()
        y.random_(0, n_class)
        
        with torch.no_grad():
            x_fake = G(z, G.shared(y))
        if not args.w_class:
            y = None
        with torch.no_grad():
            critic_val, _ = D(x_fake, y)
        critics.append(critic_val.detach().cpu())

    critics = torch.cat(critics, dim=0).view(-1).numpy()
    plt.hist(critics, num_bin, alpha = 0.5)
    plt.legend(['real', 'fake'])
    plt.show()


def make_inplace_false(m):
    if 'inplace' in m.__dict__:
        m.inplace = False 


def wip(G, D, config, args, dev):
    G.apply(make_inplace_false)
    # print(G)


def sampling(G, D, config, args, dev): 
    # Make Eval
    G.eval().to(dev)
    num_iter = 0

    # Sample Latent
    with torch.no_grad():
        for i in tqdm(range(50000 // args.size_batch)): 
            z = torch.zeros((args.size_batch, G.dim_z)).to(dev)
            z.normal_(mean=0, std=0.8)
            y = torch.ones(args.size_batch).to(dev).long() * 15
            fakes = G(z, G.shared(y))

            for fake in fakes: 
                num_iter += 1
                fake = (fake + 1) / 2
                img = ToPILImage()(fake)
                img.save('./sampling/%05d.jpeg' % num_iter)


    if True:
        pass
    if False:
        grid = make_grid(fake, nrow=2, normalize=True)
        img = ToPILImage()(grid)
        img.show()


def main():
    args = parse_args()
    dev = args.device 

    # Load Configuratuion
    with open(args.path_config, 'rb') as f:
        config = pickle.load(f)


    if args.print_config:
        for i in config:
            print(i, ':', config[i])

    # Load model
    G = models.Generator(**config)
    D = models.Discriminator(**config)

    # Load ckpt
    if args.use_pretrained_g:
        G.load_state_dict(torch.load(args.path_ckpt_g), strict=False)
    if args.use_pretrained_d:
        D.load_state_dict(torch.load(args.path_ckpt_d), strict=False)

    # Start Program
    if args.mode == 'sampling':
        print("# Sampling Started")
        sampling(G, D, config, args, dev)
    elif args.mode == 'critic':
        print("# Critic Started")
        critic(G, D, config, args, dev)
    elif args.mode == 'wip':
        wip(G, D, config, args, dev)


if __name__ == '__main__':
    main()
