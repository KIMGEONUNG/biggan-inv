import models
from os.path import join 
from encoders import EncoderFZ_16_Multi

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
    parser.add_argument('--task_name', default='encoder_fz_self_v3')
    parser.add_argument('--detail', 
        default='multi level mse')

    # IO
    parser.add_argument('--path_config', default='config.pickle')
    parser.add_argument('--path_ckpt', default='./ckpts')
    parser.add_argument('--path_ckpt_g', default='./pretrained/G_ema_256.pth')
    parser.add_argument('--path_ckpt_d', default='./pretrained/D_256.pth')
    parser.add_argument('--path_imgnet_train', default='./imgnet/train')
    parser.add_argument('--path_imgnet_val', default='./imgnet/val')
    parser.add_argument('--path_sampling', default='./sampling')
    
    # Encoder Traning
    parser.add_argument('--path_dataset_encoder', default='./dataset_encoder/')
    parser.add_argument('--path_dataset_encoder_val', default='./dataset_encoder_val/')
    parser.add_argument('--class_index', default=15)
    parser.add_argument('--num_layer', default=2)

    parser.add_argument('--num_iter', default=80000)
    parser.add_argument('--interval_save_loss', default=4)
    parser.add_argument('--interval_save_train', default=20)
    parser.add_argument('--interval_save_test', default=100)
    parser.add_argument('--interval_save_ckpt', default=100)

    # Discriminator Options
    parser.add_argument('--z_std', default=0.9)

    # Optimizer
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--b1", type=float, default=0.0)
    parser.add_argument("--b2", type=float, default=0.999)

    # Verbose
    parser.add_argument('--print_config', default=False)
    parser.add_argument('--print_generator', default=True)
    parser.add_argument('--print_discriminator', default=True)

    # loader
    parser.add_argument('--use_pretrained_g', default=True)
    parser.add_argument('--use_pretrained_d', default=False)

    # Loss
    parser.add_argument('--loss_mse_z', action='store_true', default=True)
    parser.add_argument('--loss_mse_f', action='store_true', default=True)

    # Loss coef
    parser.add_argument('--coef_mse_z', type=float, default=0.5)
    parser.add_argument('--coef_mse_f', type=float, default=1.0)

    # Others
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size_batch', default=16)
    parser.add_argument('--w_class', default=False)
    parser.add_argument('--device', default='cuda:0')

    return parser.parse_args()


class VGG16Perceptual():

    def __init__(self,
            resize=True,
            normalized_input=True,
            dev='cuda'):

        self.model = torch.hub.load('pytorch/vision:v0.8.2', 'vgg16',
                pretrained=True).to(dev).eval()

        self.normalized_intput = normalized_input
        self.dev = dev
        self.idx_targets = [1, 2, 13, 20]

        preprocess = []
        if resize:
            preprocess.append(transforms.Resize(256))
            preprocess.append(transforms.CenterCrop(224))
        if normalized_input:
            preprocess.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]))

        self.preprocess = transforms.Compose(preprocess)

    def get_mid_feats(self, x):
        x = self.preprocess(x)
        feats = []
        for i, layer in enumerate(self.model.features[:max(self.idx_targets) + 1]):
            x = layer(x)
            if i in self.idx_targets:
                feats.append(x)

        return feats

    def perceptual_loss(self, x1, x2):
        x1_feats = self.preprocess(x1)
        x2_feats = self.preprocess(x2)

        loss = 0
        for feat1, feat2 in zip(x1_feats, x2_feats):
            loss += feat1.sub(feat2).pow(2).mean()

        return loss / len(self.idx_targets)


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss


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


def get_inf_batch(loader):
    while True:
        for x in loader:
            yield x


def train(G, D, config, args, dev):
    # Make Eval
    G.eval().to(dev)
    print(args)
    if args.seed >= 0:
        set_seed(args.seed)

    # Logger
    path_log = 'runs_self/' + args.task_name
    writer = SummaryWriter(path_log)
    writer.add_text('config', str(args))
    print('logger name:', path_log)

    # Datasets
    prep = transforms.Compose([
            ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            ])
    dataset = ImageFolder(args.path_dataset_encoder, transform=prep)
    dataloader = DataLoader(dataset, batch_size=args.size_batch, shuffle=False,
            num_workers=8, drop_last=True)

    x_dataset, _ = next(iter(dataloader))
    x_dataset = x_dataset.to(dev)

    # Models 
    vgg_per = VGG16Perceptual()
    encoder = EncoderFZ_16_Multi().to(dev)

    # Optimizer
    optimizer = optim.Adam(encoder.parameters(),
            lr=args.lr, betas=(args.b1, args.b2))

    for i in tqdm(range(args.num_iter)):
        # Sample C 
        c = torch.ones(args.size_batch) * args.class_index
        c = c.to(dev).long()

        # Sample Z
        z = torch.zeros((args.size_batch, G.dim_z)).to(dev)
        z.normal_(mean=0, std=args.z_std)

        # Generate
        f = G.forward_to(z, G.shared(c), args.num_layer)
        x, fs = G.forward_from_multi(z, G.shared(c), args.num_layer, f)
        x = (x + 1) * 0.5

        # Inference
        x_gray = transforms.Grayscale()(x)
        fs_hat, z_hat = encoder(x_gray)
        f_hat = fs_hat[-1]
        fs_hat.reverse()

        # Loss
        optimizer.zero_grad()
        loss = 0
        if args.loss_mse_z:
            loss_mse_z = nn.MSELoss()(z, z_hat) 
            loss += loss_mse_z * args.coef_mse_z

        if args.loss_mse_f:
            loss_mse_f1 = nn.MSELoss()(fs[0], fs_hat[0])
            loss_mse_f2 = nn.MSELoss()(fs[1], fs_hat[1])
            loss_mse_f3 = nn.MSELoss()(fs[2], fs_hat[2])
            loss_mse_f = loss_mse_f1 + loss_mse_f2 * 0.5 + loss_mse_f3 * 0.5
            loss += loss_mse_f

        loss.backward()
        optimizer.step()

        if i % args.interval_save_train == 0:
            writer.add_scalar('mse_z', loss_mse_z.item(), i)
            writer.add_scalar('mse_f', loss_mse_f1.item(), i)
            writer.add_scalar('mse_f_total', loss_mse_f.item(), i)
        if i % args.interval_save_train == 0:
            # sample vs recon
            with torch.no_grad():
                recon = G.forward_from(z_hat, G.shared(c), 
                        args.num_layer, f_hat)
            recon = (recon + 1) * 0.5
            grid = torch.cat([x, recon], dim=0)
            grid = make_grid(grid, nrow=args.size_batch)
            writer.add_image('recon_train', grid, i)

            mse_recon_train = nn.MSELoss()(x, recon)
            writer.add_scalars('mse_recon',
                    {'train': mse_recon_train.item()}, i)

            writer.flush()
        if i % args.interval_save_test == 0:
            # use real image 
            x_gray = transforms.Grayscale()(x_dataset)
            with torch.no_grad():
                fs_hat, z_hat = encoder(x_gray)
                f_hat = fs_hat[-1]
                recon = G.forward_from(z_hat, G.shared(c), 
                        args.num_layer, f_hat)
            recon = (recon + 1) * 0.5
            grid = torch.cat([x_dataset, recon], dim=0)
            grid = make_grid(grid, nrow=args.size_batch)
            writer.add_image('recon_test', grid, i)

            mse_recon_test = nn.MSELoss()(x_dataset, recon)
            writer.add_scalars('mse_recon',
                    {'test': mse_recon_test.item()}, i)
            writer.flush()
        if i % args.interval_save_ckpt == 0:
            path_target = join(args.path_ckpt, args.task_name + '.ckpt')
            torch.save(encoder.state_dict(), path_target) 




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
        print("#@ LOAD PRETRAINED GENERATOR")
        G.load_state_dict(torch.load(args.path_ckpt_g), strict=False)
    if args.use_pretrained_d:
        print("#@ LOAD PRETRAINED DISCRIMINATOR")
        D.load_state_dict(torch.load(args.path_ckpt_d), strict=False)

    train(G, D, config, args, dev)


if __name__ == '__main__':
    main()
