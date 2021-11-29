import os
from os.path import join, exists
import models
from encoders import EncoderF_16, EncoderZ

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
    parser.add_argument('--task_name', default='encoder_fz_16_finetune_v1')
    parser.add_argument('--with z encoder', 
        default='fix the bug')

    # Mode
    parser.add_argument('--mode', default='train', 
            choices=['sampling', 'critic', 'wip', 'train'])
    
    parser.add_argument('--norm_type', default='layer', 
            choices=['instance', 'batch', 'layer'])

    # IO
    parser.add_argument('--path_ckpts', default='ckpts')
    parser.add_argument('--path_config', default='config.pickle')
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
    parser.add_argument('--interval_save_test', default=200)
    parser.add_argument('--interval_save_ckpt', default=600)

    parser.add_argument('--finetune_g', default=True)
    parser.add_argument('--finetune_d', default=True)

    # Discriminator Options
    parser.add_argument('--num_dis', default=1)
    parser.add_argument('--use_sampling_reality', default=True)

    # Optimizer
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--b1", type=float, default=0.0)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--lr_d", type=float, default=0.00003)
    parser.add_argument("--b1_d", type=float, default=0.0)
    parser.add_argument("--b2_d", type=float, default=0.999)

    # Verbose
    parser.add_argument('--print_config', default=False)
    parser.add_argument('--print_generator', default=True)
    parser.add_argument('--print_discriminator', default=True)
    parser.add_argument('--print_encoder', default=True)

    # loader
    parser.add_argument('--use_pretrained_g', default=True)
    parser.add_argument('--use_pretrained_d', default=True)
    parser.add_argument('--real_target', default='dataset',
            choices=['sample', 'dataset'])

    # Loss
    parser.add_argument('--loss_mse', action='store_true', default=True)
    parser.add_argument('--loss_lpips', action='store_true', default=True)
    parser.add_argument('--loss_reg', action='store_true', default=True)
    parser.add_argument('--loss_adv', action='store_true', default=True)

    # Loss coef
    parser.add_argument('--coef_mse', type=float, default=1.0)
    parser.add_argument('--coef_lpips', type=float, default=0.1)
    parser.add_argument('--coef_adv', type=float, default=0.03)
    parser.add_argument('--coef_reg', type=float, default=0.0125)

    # Others
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size_batch', default=8)
    parser.add_argument('--w_class', default=False)
    parser.add_argument('--device', default='cuda:0')

    return parser.parse_args()


class VGG16Perceptual():

    def __init__(self,
            resize=True,
            normalized_input=True,
            dev='cuda',
            load_pickle=True):

        if load_pickle:
            import pickle 
            with open('./vgg16.pickle', 'rb') as f:
                self.model = pickle.load(f).to(dev).eval()
        else:
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
    if(args.finetune_g):
        print("# GENERATOR FINETUNE")
        G.train().to(dev)
    else:
        print("# GENERATOR FIX")
        G.eval().to(dev)

    if(args.finetune_d):
        print("# DISCRIMINATOR FINETUNE")
        D.train().to(dev)
    else:
        print("# DISCRIMINATOR FIX")
        D.eval().to(dev)

    print(args)
    if args.seed >= 0:
        set_seed(args.seed)

    # Make directory for checkpoints    
    if not exists(args.path_ckpts):
        os.mkdir(args.path_ckpts)
    path_ckpts = join(args.path_ckpts, args.task_name)
    if not exists(path_ckpts):
        os.mkdir(path_ckpts)
       
    # Save arguments
    with open(join(path_ckpts, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Logger
    path_log = 'runs_finetune/' + args.task_name
    writer = SummaryWriter(path_log)
    writer.add_text('config', str(args))
    print('logger name:', path_log)

    # Models 
    vgg_per = VGG16Perceptual()
    encoder_f = EncoderF_16(norm=args.norm_type).to(dev)
    encoder_z = EncoderZ(ch_out=G.dim_z, norm=args.norm_type).to(dev)
    encoder_f.train()
    encoder_z.train()

    if args.print_encoder:
        print(encoder_f)
        print(encoder_z)

    # Optimizer
    params = list(encoder_f.parameters())
    params += list(encoder_z.parameters())
    if args.finetune_g:
        params += list(G.parameters())
    optimizer_g = optim.Adam(params,
            lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d = optim.Adam(D.parameters(),
            lr=args.lr_d, betas=(args.b1_d, args.b2_d))

    # Datasets
    prep = transforms.Compose([
            ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            ])

    if args.real_target == 'sample':
        dataset_real = ImageFolder(args.path_sampling, transform=prep)
    elif args.real_target == 'dataset':
        dataset_real = ImageFolder(args.path_dataset_encoder, transform=prep)
    dataloader_real = DataLoader(dataset_real, batch_size=args.size_batch, shuffle=False,
            num_workers=8, drop_last=True)
    dataloader_real = get_inf_batch(dataloader_real)

    # Fix test samples
    with torch.no_grad():
        dataset = ImageFolder(args.path_dataset_encoder, transform=prep)
        dataloader = DataLoader(dataset, batch_size=args.size_batch, shuffle=False,
                num_workers=8, drop_last=True)

        dataset_val = ImageFolder(args.path_dataset_encoder_val, transform=prep)
        dataloader_val = DataLoader(dataset_val, batch_size=args.size_batch, shuffle=False,
                num_workers=8, drop_last=True)

        x_test_val, _ = next(iter(dataloader_val))
        x_test_val = transforms.Grayscale()(x_test_val)

        x_test, _ = next(iter(dataloader))
        grid_init = make_grid(x_test, nrow=4)
        writer.add_image('GT', grid_init)
        writer.flush()

        c_test = torch.ones(args.size_batch) * args.class_index
        c_test = c_test.to(dev).long()
        c_test = G.shared(c_test)

        x_test = transforms.Grayscale()(x_test)


    dataset = ImageFolder(args.path_dataset_encoder, transform=prep)
    dataloader = DataLoader(dataset, batch_size=args.size_batch, shuffle=True,
            num_workers=8, drop_last=True)


    num_iter = 0
    for epoch in range(args.num_epoch):
        for i, (x, _) in enumerate(tqdm(dataloader)):
            x = x.to(dev)
            x_gray = transforms.Grayscale()(x)

            c = torch.ones(args.size_batch) * args.class_index
            c = c.to(dev).long()

            # Discriminator Loss
            if args.loss_adv: 
                for _ in range(args.num_dis):
                    # Infer f
                    f = encoder_f(x_gray) # [batch, 1024, 16, 16]
                    z = encoder_z(x_gray) # [batch, 1024, 16, 16]
                    fake = G.forward_from(z, G.shared(c), args.num_layer, f)

                    optimizer_d.zero_grad()

                    x_real, _ = dataloader_real.__next__()
                    x_real = x_real.to(dev)
                    x_real = (x_real - 0.5) * 2

                    critic_real, _ = D(x_real, c)
                    critic_fake, _ = D(fake, c)
                    d_loss_real, d_loss_fake = loss_hinge_dis(critic_fake, critic_real)
                    loss_d = (d_loss_real + d_loss_fake) / 2  

                    loss_d.backward()
                    optimizer_d.step()

            # Generator Loss 
            f = encoder_f(x_gray) # [batch, 1024, 16, 16]
            z = encoder_z(x_gray) # [batch, 1024, 16, 16]
            fake = G.forward_from(z, G.shared(c), args.num_layer, f)

            optimizer_g.zero_grad()
            if args.loss_adv:
                critic, _ = D(fake, c)
                loss_g = loss_hinge_gen(critic) * args.coef_adv
                loss_g.backward(retain_graph=True)

            fake = fake.add(1).div(2)
            if args.loss_mse:
                loss_mse = args.coef_mse * nn.MSELoss()(x, fake)
                # loss_mse.backward(retain_graph=True)
            if args.loss_lpips:
                loss_lpips = args.coef_lpips * vgg_per.perceptual_loss(x, fake)
                # loss_lpips.backward(retain_graph=True)
            if args.loss_reg:
                loss_reg = z.norm(2) * 0.5 * args.coef_reg
                # loss_reg.backward()
            loss = loss_mse + loss_lpips + loss_reg
            loss.backward()

            optimizer_g.step()

            # Logger
            if num_iter % args.interval_save_loss == 0:
                writer.add_scalar('mse', loss_mse.item(), num_iter)
                writer.add_scalar('lpips', loss_lpips.item(), num_iter)
                if args.loss_adv:
                    writer.add_scalars('GAN loss', 
                        {'G': loss_g.item(), 'D': loss_d.item()}, num_iter)

            if num_iter % args.interval_save_train == 0:
                with torch.no_grad():
                    f = encoder_f(x_test.to(dev))
                    z_test = encoder_z(x_test.to(dev))
                    output = G.forward_from(z_test, c_test, args.num_layer, f)
                    output = output.add(1).div(2)
                    grid = make_grid(output, nrow=4)
                    writer.add_image('recon_train', grid, num_iter)
                    writer.flush()

            if num_iter % args.interval_save_test == 0:
                with torch.no_grad():
                    f = encoder_f(x_test_val.to(dev))
                    z_text = encoder_z(x_test_val.to(dev))
                    output = G.forward_from(z_test, c_test, args.num_layer, f)
                    output = output.add(1).div(2)
                    grid = make_grid(output, nrow=4)
                    writer.add_image('recon_val', grid, num_iter)
                    writer.flush()

            if num_iter % args.interval_save_ckpt == 0:
                if args.finetune_g:
                    name = 'G_%07d.ckpt' % num_iter 
                    path = join(path_ckpts, name) 
                    torch.save(G.state_dict(), path) 
                if args.finetune_d:
                    name = 'D_%07d.ckpt' % num_iter 
                    path = join(path_ckpts, name) 
                    torch.save(D.state_dict(), path) 
                name = 'EF_%07d.ckpt' % num_iter 
                path = join(path_ckpts, name) 
                torch.save(encoder_f.state_dict(), path) 
                name = 'EZ_%07d.ckpt' % num_iter 
                path = join(path_ckpts, name) 
                torch.save(encoder_z.state_dict(), path) 

            num_iter += 1


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
