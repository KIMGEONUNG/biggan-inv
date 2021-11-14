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
    parser.add_argument('--task_name', default='encoder_fz_self_transfer_v3')
    parser.add_argument('--detail', 
        default='only use lpips')

    # IO
    parser.add_argument('--path_config', default='config.pickle')
    parser.add_argument('--path_ckpt', default='./ckpts')
    parser.add_argument('--path_ckpt_target', default='./ckpts/encoder_fz_self_v3.ckpt')
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

    parser.add_argument('--num_epoch', default=400)
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
    parser.add_argument('--loss_mse', action='store_true', default=True)
    parser.add_argument('--loss_lpips', action='store_true', default=False)

    # Loss coef
    parser.add_argument('--coef_mse', type=float, default=1.0)
    parser.add_argument('--coef_lpips', type=float, default=0.1)

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
    path_log = 'runs_self_transfer/' + args.task_name
    writer = SummaryWriter(path_log)
    writer.add_text('config', str(args))
    print('logger name:', path_log)

    # Datasets for logger
    prep = transforms.Compose([
            ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            ])

    dataset = ImageFolder(args.path_dataset_encoder, transform=prep)
    dataloader = DataLoader(dataset, batch_size=args.size_batch, shuffle=False,
            num_workers=8, drop_last=True)
    x_test, _ = next(iter(dataloader))
    x_test_gray = transforms.Grayscale()(x_test)

    dataset_val = ImageFolder(args.path_dataset_encoder_val, transform=prep)
    dataloader_val = DataLoader(dataset_val, batch_size=args.size_batch, shuffle=False,
            num_workers=8, drop_last=True)
    x_test_val, _ = next(iter(dataloader_val))
    x_test_val_gray = transforms.Grayscale()(x_test_val)

    grid_init = make_grid(x_test, nrow=4)
    writer.add_image('GT_train', grid_init)
    writer.flush()
    grid_init = make_grid(x_test_val, nrow=4)
    writer.add_image('GT_val', grid_init)
    writer.flush()

    dataset = ImageFolder(args.path_dataset_encoder, transform=prep)
    dataloader = DataLoader(dataset, batch_size=args.size_batch, shuffle=True,
            num_workers=8, drop_last=True)

    # Models 
    vgg_per = VGG16Perceptual()
    encoder = EncoderFZ_16_Multi().to(dev)
    encoder.load_state_dict(torch.load(args.path_ckpt_target))

    # Optimizer
    optimizer = optim.Adam(encoder.parameters(),
            lr=args.lr, betas=(args.b1, args.b2))

    num_iter = 0 
    for epoch in range(args.num_epoch):
        for i, (x, _) in enumerate(tqdm(dataloader)):
            num_iter += 1

            # Define inputs
            x = x.to(dev)
            x_gray = transforms.Grayscale()(x)

            # Sample C 
            c = torch.ones(args.size_batch) * args.class_index
            c = c.to(dev).long()

            # Inference
            fs_hat, z_hat = encoder(x_gray)
            f_hat = fs_hat[-1]
            recon = G.forward_from(z_hat, G.shared(c), 
                    args.num_layer, f_hat)

            # Loss
            optimizer.zero_grad()
            loss = 0
            if args.loss_mse:
                x_down = transforms.Resize(64)(x)
                recon_down = transforms.Resize(64)(recon)
                loss_mse = nn.MSELoss()(x_down, recon_down) 
                loss += loss_mse * args.coef_mse

            if args.loss_lpips:
                loss_lpips = vgg_per.perceptual_loss(x, recon)
                loss += loss_lpips * args.coef_lpips

            loss.backward()
            optimizer.step()

            if i % args.interval_save_train == 0:
                if args.loss_mse:
                    writer.add_scalar('mse', loss_mse.item(), num_iter)
                if args.loss_lpips:
                    writer.add_scalar('lpips', loss_lpips.item(), num_iter)
                
            if i % args.interval_save_train == 0:
                with torch.no_grad():
                    fs_hat, z_hat = encoder(x_test_gray.to(dev))
                    f_hat = fs_hat[-1]
                    output = G.forward_from(z_hat, G.shared(c), 
                            args.num_layer, f_hat)
                    output = output.add(1).div(2)
                grid = make_grid(output, nrow=4)
                writer.add_image('recon_train', grid, num_iter)
                writer.flush()

            if i % args.interval_save_test == 0:
                with torch.no_grad():
                    fs_hat, z_hat = encoder(x_test_val_gray.to(dev))
                    f_hat = fs_hat[-1]
                    output = G.forward_from(z_hat, G.shared(c), 
                            args.num_layer, f_hat)
                    output = output.add(1).div(2)
                grid = make_grid(output, nrow=4)
                writer.add_image('recon_val', grid, num_iter)
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
