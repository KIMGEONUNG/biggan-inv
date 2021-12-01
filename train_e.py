import os
import numpy as np
from skimage import color
from os.path import join, exists
import models
from encoders import EncoderF_16

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader 
from torch.utils.data import Subset 

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
    parser.add_argument('--task_name', default='test4')
    parser.add_argument('--detail', default='train eval seperation')

    # Mode
    parser.add_argument('--use_z_encoder', default=True)

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

    # Encoder Traning
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

    # Loss
    parser.add_argument('--loss_mse', action='store_true', default=True)
    parser.add_argument('--loss_lpips', action='store_true', default=True)
    parser.add_argument('--loss_adv', action='store_true', default=True)

    # Loss coef
    parser.add_argument('--coef_mse', type=float, default=1.0)
    parser.add_argument('--coef_lpips', type=float, default=0.1)
    parser.add_argument('--coef_adv', type=float, default=0.03)

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


def extract(dataset, target_ids):
    '''
    extract data element based on class index
    '''
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in target_ids:
            indices.append(i)
    return Subset(dataset, indices)


def prepare_dataset(
        path_train,
        path_valid,
        index_target,
        prep=transforms.Compose([
            ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            ])):


    dataset = ImageFolder(path_train, transform=prep)
    dataset = extract(dataset, index_target)

    dataset_val = ImageFolder(path_valid, transform=prep)
    dataset_val = extract(dataset_val, index_target)
    return dataset, dataset_val


def extract_sample(dataset, size_batch, num_iter, is_shuffle):
    dataloader = DataLoader(dataset, batch_size=size_batch,
            shuffle=is_shuffle, num_workers=4,
            drop_last=True)
    xs = []
    xgs = []
    cs = []
    for i, (x, c) in enumerate(dataloader):
        if i >= num_iter:
            break
        xg = transforms.Grayscale()(x)
        xs.append(x), cs.append(c), xgs.append(xg)
    return {'xs': xs, 'cs': cs, 'xs_gray': xgs}


def lab_fusion(x_l, x_ab):
    labs = []
    for img_gt, img_hat in zip(x_l, x_ab):

        img_gt = img_gt.permute(1, 2, 0)
        img_hat = img_hat.permute(1, 2, 0)

        img_gt = color.rgb2lab(img_gt)
        img_hat = color.rgb2lab(img_hat)
        
        l = img_gt[:, :, :1]
        ab = img_hat[:, :, 1:]
        img_fusion = np.concatenate((l, ab), axis=-1)
        img_fusion = color.lab2rgb(img_fusion)
        img_fusion = torch.from_numpy(img_fusion)
        img_fusion = img_fusion.permute(2, 0, 1)
        labs.append(img_fusion)
    labs = torch.stack(labs)
     
    return labs


def make_grid_multi(xs, nrow=4):
    return make_grid(torch.cat(xs, dim=0), nrow=nrow)


def train(G, D, config, args, dev,
            dataset=None,
            sample_train=None,
            sample_valid=None,
            writer=None,
            path_ckpts=None,
        ):
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


    # Models 
    vgg_per = VGG16Perceptual()
    encoder = EncoderF_16(norm=args.norm_type).to(dev)
    if args.print_encoder:
        print(encoder)

    # Optimizer
    params = list(encoder.parameters())
    if args.finetune_g:
        params += list(G.parameters())
    optimizer_g = optim.Adam(params,
            lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d = optim.Adam(D.parameters(),
            lr=args.lr_d, betas=(args.b1_d, args.b2_d))

    # Datasets
    dataloader_real = DataLoader(dataset, batch_size=args.size_batch, shuffle=True,
            num_workers=args.num_worker, drop_last=True)
    dataloader_real = get_inf_batch(dataloader_real)
    dataloader = DataLoader(dataset, batch_size=args.size_batch, shuffle=True,
            num_workers=args.num_worker, drop_last=True)

    num_iter = 0
    for epoch in range(args.num_epoch):
        for i, (x, c) in enumerate(tqdm(dataloader)):
            G.train()
            encoder.train()

            x = x.to(dev)
            c = c.to(dev)
            x_gray = transforms.Grayscale()(x)

            # Sample z
            z = torch.zeros((args.size_batch, G.dim_z)).to(dev)
            z.normal_(mean=0, std=0.8)


            # Discriminator Loss
            if args.loss_adv: 
                for _ in range(args.num_dis):
                    # Infer f
                    f = encoder(x_gray) # [batch, 1024, 16, 16]
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
            f = encoder(x_gray) # [batch, 1024, 16, 16]
            fake = G.forward_from(z, G.shared(c), args.num_layer, f)

            optimizer_g.zero_grad()
            if args.loss_adv:
                critic, _ = D(fake, c)
                loss_g = loss_hinge_gen(critic) * args.coef_adv
                loss_g.backward(retain_graph=True)

            fake = fake.add(1).div(2)
            if args.loss_mse:
                loss_mse = args.coef_mse * nn.MSELoss()(x, fake)
                loss_mse.backward(retain_graph=True)
            if args.loss_lpips:
                loss_lpips = args.coef_lpips * vgg_per.perceptual_loss(x, fake)
                loss_lpips.backward()

            optimizer_g.step()

            # Logger
            if num_iter % args.interval_save_loss == 0:
                writer.add_scalar('mse', loss_mse.item(), num_iter)
                writer.add_scalar('lpips', loss_lpips.item(), num_iter)
                if args.loss_adv:
                    writer.add_scalars('GAN loss', 
                        {'G': loss_g.item(), 'D': loss_d.item()}, num_iter)

            if num_iter % args.interval_save_train == 0:
                G.eval()
                encoder.eval()

                outputs_rgb = []
                outputs_fusion = []
                with torch.no_grad():
                    for id_sample in range(len(sample_valid['xs'])):
                        z = torch.zeros((args.size_batch, G.dim_z))
                        z.normal_(mean=0, std=0.8)
                        x_gt = sample_train['xs'][id_sample]

                        x = sample_train['xs_gray'][id_sample]
                        c = sample_train['cs'][id_sample]
                        z, x, c = z.to(dev), x.to(dev), c.to(dev)

                        f = encoder(x)
                        output = G.forward_from(z, G.shared(c), args.num_layer, f)
                        output = output.add(1).div(2).detach().cpu()
                        output_fusion = lab_fusion(x_gt, output)
                        outputs_rgb.append(output)
                        outputs_fusion.append(output_fusion)

                grid = make_grid_multi(outputs_rgb, nrow=4)
                writer.add_image('recon_train_rgb', grid, num_iter)

                grid = make_grid_multi(outputs_fusion, nrow=4)
                writer.add_image('recon_train_fusion', grid, num_iter)

                writer.flush()

            if num_iter % args.interval_save_test == 0:
                G.eval()
                encoder.eval()
                outputs_rgb = []
                outputs_fusion = []
                with torch.no_grad():
                    for id_sample in range(len(sample_valid['xs'])):
                        z = torch.zeros((args.size_batch, G.dim_z))
                        z.normal_(mean=0, std=0.8)
                        x_gt = sample_valid['xs'][id_sample]

                        x = sample_valid['xs_gray'][id_sample]
                        c = sample_valid['cs'][id_sample]
                        z, x, c = z.to(dev), x.to(dev), c.to(dev)

                        f = encoder(x)
                        output = G.forward_from(z, G.shared(c), args.num_layer, f)
                        output = output.add(1).div(2).detach().cpu()
                        output_fusion = lab_fusion(x_gt, output)
                        outputs_rgb.append(output)
                        outputs_fusion.append(output_fusion)

                grid = make_grid_multi(outputs_rgb, nrow=4)
                writer.add_image('recon_valid_rgb', grid, num_iter)

                grid = make_grid_multi(outputs_fusion, nrow=4)
                writer.add_image('recon_valid_fusion', grid, num_iter)

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
                name = 'E_%07d.ckpt' % num_iter 
                path = join(path_ckpts, name) 
                torch.save(encoder.state_dict(), path) 

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
    prep = transforms.Compose([
            ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            ])

    # DATASETS
    dataset, dataset_val = prepare_dataset(
            args.path_imgnet_train,
            args.path_imgnet_val,
            args.index_target,
            prep=prep)

    is_shuffle = False 
    sample_train = extract_sample(dataset, args.size_batch, args.iter_sample, is_shuffle)
    sample_valid = extract_sample(dataset_val, args.size_batch, args.iter_sample, is_shuffle)

    grid_init = make_grid_multi(sample_train['xs'], nrow=4)
    writer.add_image('GT_train', grid_init)
    grid_init = make_grid_multi(sample_valid['xs'], nrow=4)
    writer.add_image('GT_valid', grid_init)
    writer.flush()

    train(G, D, config, args, dev, 
            dataset=dataset,
            sample_train=sample_train,
            sample_valid=sample_valid,
            writer=writer,
            path_ckpts=path_ckpts,
            )


if __name__ == '__main__':
    main()
