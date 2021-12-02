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
from torch.utils.data.distributed import DistributedSampler

import pickle
import argparse
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchvision.transforms import (ToPILImage, Compose, ToTensor,
        Resize, CenterCrop)
from tqdm import tqdm


import torch.multiprocessing as mp
import torch.distributed as dist 
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
    parser.add_argument('--task_name', default='test9')
    parser.add_argument('--detail', default='multi gpu')

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
    parser.add_argument('--dim_z', type=int, default=119)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size_batch', default=4)
    parser.add_argument('--w_class', default=False)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--multi_gpu', default=True)

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


def setup_dist(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


class Colorizer(nn.Module):
    def __init__(self, config, path_ckpt_g, norm_type,
            id_mid_layer =2):
        super().__init__()
        self.id_mid_layer = id_mid_layer  
        self.E = EncoderF_16(norm=norm_type)
        self.G = models.Generator(**config)
        self.G.load_state_dict(torch.load(path_ckpt_g), strict=False)

    def forward(self, x_gray, c, z):
        f = self.E(x_gray) 
        output = self.G.forward_from(z, self.G.shared(c), 
                self.id_mid_layer, f)
        return output


def train(dev, world_size, config, args,
            dataset=None,
            sample_train=None,
            sample_valid=None,
            path_ckpts=None,
            path_log=None,
        ):
    writer = SummaryWriter(path_log)
    use_multi_gpu = world_size > 1

    if use_multi_gpu:
        setup_dist(dev, world_size)

    # Setup model
    EG = Colorizer(config, args.path_ckpt_g, args.norm_type,
            id_mid_layer=args.num_layer)
    EG.train()

    D = models.Discriminator(**config)
    D.train()
    if args.use_pretrained_d:
        D.load_state_dict(torch.load(args.path_ckpt_d), strict=False)

    # Load model
    vgg_per = VGG16Perceptual(dev=dev)
    EG = EG.to(dev)
    D = D.to(dev)
    if use_multi_gpu:
        EG = DDP(EG, device_ids=[dev], find_unused_parameters=True)
        D = DDP(D, device_ids=[dev], find_unused_parameters=True)

    # Optimizer
    optimizer_g = optim.Adam(EG.parameters(),
            lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d = optim.Adam(D.parameters(),
            lr=args.lr_d, betas=(args.b1_d, args.b2_d))

    # Datasets
    sampler, sampler_real = None, None
    if use_multi_gpu:
        sampler = DistributedSampler(dataset)
        sampler_real = DistributedSampler(dataset)

    dataloader_real = DataLoader(dataset, batch_size=args.size_batch, 
            shuffle=True if sampler_real is None else False, 
            sampler=sampler_real,
            num_workers=args.num_worker, drop_last=True)
    dataloader_real = get_inf_batch(dataloader_real)

    dataloader = DataLoader(dataset, batch_size=args.size_batch, 
            shuffle=True if sampler is None else False, 
            sampler=sampler,
            num_workers=args.num_worker, drop_last=True)

    num_iter = 0
    for epoch in range(args.num_epoch):
        if use_multi_gpu:
            sampler.set_epoch(epoch)
        for i, (x, c) in enumerate(tqdm(dataloader)):
            EG.train()

            x = x.to(dev)
            c = c.to(dev)
            x_gray = transforms.Grayscale()(x)

            # Sample z
            z = torch.zeros((args.size_batch, args.dim_z)).to(dev)
            z.normal_(mean=0, std=0.8)

            # Discriminator Loss
            if args.loss_adv: 
                for _ in range(args.num_dis):
                    with torch.no_grad():
                        fake = EG(x_gray, c, z)

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
            fake = EG(x_gray, c, z)

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
            condition = dev == 0 if use_multi_gpu else True
            if num_iter % args.interval_save_loss == 0 and condition:
                make_log_scalar(writer, num_iter, loss_mse, loss_lpips, loss_d, loss_g)

            if num_iter % args.interval_save_train == 0 and condition:
                make_log_img(EG, args.dim_z, writer, args, sample_train,
                        dev, num_iter, 'train')
            if num_iter % args.interval_save_test == 0 and condition:
                make_log_img(EG, args.dim_z, writer, args, sample_valid,
                        dev, num_iter, 'valid')
            if num_iter % args.interval_save_ckpt == 0 and condition:
                if use_multi_gpu:
                    make_log_ckpt(EG.module, D.module,
                            args, num_iter, path_ckpts)
                else:
                    make_log_ckpt(EG, D, args, num_iter, path_ckpts)

            num_iter += 1


def make_log_ckpt(EG, D, args, num_iter, path_ckpts):

    name = 'D_%07d.ckpt' % num_iter 
    path = join(path_ckpts, name) 
    torch.save(D.state_dict(), path) 

    name = 'EG_%07d.ckpt' % num_iter 
    path = join(path_ckpts, name) 
    torch.save(EG.state_dict(), path) 


def make_log_scalar(writer, num_iter, loss_mse, loss_lpips, loss_d, loss_g):
    writer.add_scalar('mse', loss_mse.item(), num_iter)
    writer.add_scalar('lpips', loss_lpips.item(), num_iter)
    writer.add_scalars('GAN loss', 
        {'G': loss_g.item(), 'D': loss_d.item()}, num_iter)


def make_log_img(EG, dim_z, writer, args, sample, dev, num_iter, name):
    EG.eval()

    outputs_rgb = []
    outputs_fusion = []
    with torch.no_grad():
        for id_sample in range(len(sample['xs'])):
            z = torch.zeros((args.size_batch, dim_z))
            z.normal_(mean=0, std=0.8)
            x_gt = sample['xs'][id_sample]

            x = sample['xs_gray'][id_sample]
            c = sample['cs'][id_sample]
            z, x, c = z.to(dev), x.to(dev), c.to(dev)

            output = EG(x, c, z)
            output = output.add(1).div(2).detach().cpu()
            output_fusion = lab_fusion(x_gt, output)
            outputs_rgb.append(output)
            outputs_fusion.append(output_fusion)

    grid = make_grid_multi(outputs_rgb, nrow=4)
    writer.add_image('recon_%s_rgb' % name, 
            grid, num_iter)

    grid = make_grid_multi(outputs_fusion, nrow=4)
    writer.add_image('recon_%s_fusion' % name, 
            grid, num_iter)

    writer.flush()


def main():
    args = parse_args()

    # GPU OPTIONS
    use_multi_gpu = False
    num_gpu = torch.cuda.device_count()
    if num_gpu == 0:
        raise Exception('No available GPU')
    elif num_gpu == 1 or args.multi_gpu == False:
        dev = args.device 
        print('Use single GPU:', dev)
    elif num_gpu > 1 and args.multi_gpu == True:
        use_multi_gpu = True 
        print('Use multi GPU: %02d EA' % num_gpu)
    else:
        raise Exception('Invalid GPU setting')

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

    is_shuffle = True 
    sample_train = extract_sample(dataset, args.size_batch, args.iter_sample, is_shuffle)
    sample_valid = extract_sample(dataset_val, args.size_batch, args.iter_sample, is_shuffle)

    grid_init = make_grid_multi(sample_train['xs'], nrow=4)
    writer.add_image('GT_train', grid_init)
    grid_init = make_grid_multi(sample_valid['xs'], nrow=4)
    writer.add_image('GT_valid', grid_init)
    writer.flush()
    writer.close()

    if use_multi_gpu:
        print('Use Multi_GPU')
        mp.spawn(train,
            args=(num_gpu, config, args, dataset, sample_train, sample_valid, path_ckpts, path_log),
            nprocs=num_gpu)
    else:
        train(dev, 1, config, args, 
                dataset=dataset,
                sample_train=sample_train,
                sample_valid=sample_valid,
                path_ckpts=path_ckpts,
                path_log=path_log
                )


if __name__ == '__main__':
    main()
