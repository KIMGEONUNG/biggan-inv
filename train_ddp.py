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

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

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

    # Predicate 
    parser.add_argument('--use_z_encoder', default=True)

    # IO
    parser.add_argument('--path_log', default='runs_imgnet')
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
    parser.add_argument('--interval_save_ckpt', default=1500)


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
    parser.add_argument('--size_batch', default=2)
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


class EGD(nn.Module):
    __constants__ = ['size_batch', 'use_z_encoder']

    def __init__(self, G, D, **kargs):
        super().__init__()

        # Trivial Attributes
        self.norm_type = kargs['norm_type']
        self.num_layer = kargs['num_layer']

        # Models
        self.G = G 
        self.D = D 
        self.E_F = EncoderF_16(norm=self.norm_type)
        self.E_Z = EncoderZ(ch_out=G.dim_z, norm=self.norm_type)


    def _sample_z(self, size_batch, std=0.8):
        z = torch.zeros(size_batch, self.G.dim_z)
        z.normal_(mean=0, std=std)
        return z


    def forward(self, x, c, action=None):

        if action == 'critic':
            critic = self.D(x, c)
            return critic

        if action is None:
            f = self.E_F(x) # [batch, 1024, 16, 16]
            z = self.E_Z(x) # [batch, 1024, 16, 16]
            x_hat = self.G.forward_from(z, self.G.shared(c), self.num_layer, f)

            return x_hat, f, z 
        raise


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, world_size):
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

    print('here')


def main():
    args = parse_args()

    # Load Configuratuion
    with open(args.path_config, 'rb') as f:
        config = pickle.load(f)

    if args.print_config:
        for i in config:
            print(i, ':', config[i])

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
    path_log = join(args.path_log, args.task_name) 
    writer = SummaryWriter(path_log)
    writer.add_text('config', str(args))
    print('logger name:', path_log)

    # DEFINE MODEL
    G = models.Generator(**config)
    D = models.Discriminator(**config)
    if args.use_pretrained_g:
        print("#@ LOAD PRETRAINED GENERATOR")
        G.load_state_dict(torch.load(args.path_ckpt_g), strict=False)
    if args.use_pretrained_d:
        print("#@ LOAD PRETRAINED DISCRIMINATOR")
        D.load_state_dict(torch.load(args.path_ckpt_d), strict=False)
    model = EGD(G, D, **vars(args))
    model.cuda()

    vgg_per = VGG16Perceptual()

    # OPTIMIZER
    params = list(model.E_F.parameters())
    params += list(model.E_Z.parameters())
    if args.finetune_g:
        params += list(G.parameters())
    optimizer_g = optim.Adam(params,
            lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d = optim.Adam(D.parameters(),
            lr=args.lr_d, betas=(args.b1_d, args.b2_d))

    # DATASET
    prep = transforms.Compose([
            ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            ])

    dataset = ImageFolder(args.path_imgnet_train, transform=prep)

    # Fix test samples
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=args.size_batch, 
                shuffle=True, num_workers=8, drop_last=True)
        dataset_val = ImageFolder(args.path_imgnet_val, transform=prep)
        dataloader_val = DataLoader(dataset_val, batch_size=args.size_batch, 
                shuffle=True, num_workers=8, drop_last=True)

        x_test, c_test = next(iter(dataloader))
        x_test_gray = transforms.Grayscale()(x_test)

        x_test_val, c_test_val = next(iter(dataloader_val))
        x_test_val_gray = transforms.Grayscale()(x_test_val)

        grid_init = make_grid(x_test, nrow=4)
        writer.add_image('GT_Train', grid_init)
        writer.flush()

        grid_init = make_grid(x_test_val, nrow=4)
        writer.add_image('GT_Valid', grid_init)
        writer.flush()

    dataloader_g = DataLoader(dataset, batch_size=args.size_batch, 
            shuffle=True, num_workers=1, drop_last=True)

    dataloader_d = DataLoader(dataset, batch_size=args.size_batch, 
            shuffle=True, num_workers=1, drop_last=True)
    dataloader_d = get_inf_batch(dataloader_d)


    model = nn.DataParallel(model)
    num_iter = 0
    for epoch in range(args.num_epoch):
        for i, (x, c) in enumerate(tqdm(dataloader_g)):
            model.train()
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()

            x_gray = transforms.Grayscale()(x)

            x = x.cuda()
            x_gray = x_gray.cuda()
            c = c.cuda()

            # DISCRIMINATOR LOSS
            if args.loss_adv: 
                for _ in range(args.num_dis):
                    x_fake, f, z = model(x_gray, c)

                    x_real, _ = dataloader_d.__next__()
                    x_real = x_real.cuda()
                    x_real = (x_real - 0.5) * 2

                    critic_real, _ = model(x_real, c, action='critic')
                    critic_fake, _ = model(x_fake, c, action='critic')
                    d_loss_real, d_loss_fake = loss_hinge_dis(critic_fake, critic_real)
                    loss_d = (d_loss_real + d_loss_fake) / 2  

                    loss_d.backward()
                    optimizer_d.step()

            # GENERATOR LOSS 
            x_fake, f, z = model(x_gray, c)
            if args.loss_adv:
                critic, _ = model(x_fake, c, action='critic')
                loss_g = loss_hinge_gen(critic) * args.coef_adv

            x_fake = x_fake.add(1).div(2)
            if args.loss_mse:
                loss_mse = args.coef_mse * nn.MSELoss()(x, x_fake)
            if args.loss_lpips:
                loss_lpips = args.coef_lpips * vgg_per.perceptual_loss(x, x_fake)
            if args.loss_reg:
                loss_reg = z.norm(2) * 0.5 * args.coef_reg

            loss = loss_g + loss_mse + loss_lpips + loss_reg
            loss.backward()
            optimizer_g.step()

            # LOGGING
            if num_iter % args.interval_save_loss == 0:
                writer.add_scalar('mse', loss_mse.item(), num_iter)
                writer.add_scalar('lpips', loss_lpips.item(), num_iter)
                if args.loss_adv:
                    writer.add_scalars('GAN loss', 
                        {'G': loss_g.item(), 'D': loss_d.item()}, num_iter)

            if num_iter % args.interval_save_train == 0:
                with torch.no_grad():
                    model.eval()
                    output, _, _ = model(x_test_gray, c_test)
                    output = output.add(1).div(2)
                    grid = make_grid(output, nrow=4)
                    writer.add_image('recon_train', grid, num_iter)
                    writer.flush()

            if num_iter % args.interval_save_test == 0:
                with torch.no_grad():
                    model.eval()
                    output, _, _ = model(x_test_val_gray, c_test_val)
                    output = output.add(1).div(2)
                    grid = make_grid(output, nrow=4)
                    writer.add_image('recon_val', grid, num_iter)
                    writer.flush()

            if num_iter % args.interval_save_ckpt == 0:
                name = 'EGD_%07d.ckpt' % num_iter 
                path = join(path_ckpts, name) 
                torch.save(model.state_dict(), path) 

            num_iter += 1


if __name__ == '__main__':
    # main()
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)

