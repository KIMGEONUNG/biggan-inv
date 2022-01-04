import os
from os.path import join, exists
import models
from models import Colorizer, VGG16Perceptual

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader 
from torch.utils.data.distributed import DistributedSampler

import pickle
import argparse
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm

from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.losses import loss_fn_d, loss_fn_g
from utils.common_utils import (extract_sample, lab_fusion,set_seed,
        make_grid_multi, prepare_dataset)
from utils.logger import make_log_scalar, make_log_img, make_log_ckpt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='port')
    parser.add_argument('--detail', default='mv')

    # Mode
    parser.add_argument('--norm_type', default='adabatch', 
            choices=['instance', 'batch', 'layer', 'adain', 'adabatch'])
    parser.add_argument('--activation', default='relu', 
            choices=['relu', 'lrelu', 'sigmoid'])
    parser.add_argument('--weight_init', default='ortho', 
            choices=['xavier', 'N02', 'ortho', ''])

    # IO
    parser.add_argument('--path_log', default='runs_refact')
    parser.add_argument('--path_ckpts', default='ckpts')
    parser.add_argument('--path_config', default='./pretrained/config.pickle')
    parser.add_argument('--path_vgg', default='./pretrained/vgg16.pickle')
    parser.add_argument('--path_ckpt_g', default='./pretrained/G_ema_256.pth')
    parser.add_argument('--path_ckpt_d', default='./pretrained/D_256.pth')
    parser.add_argument('--path_imgnet_train', default='./imgnet/train')
    parser.add_argument('--path_imgnet_val', default='./imgnet/val')

    parser.add_argument('--index_target',
            type=int, nargs='+', default=list(range(50)))
    parser.add_argument('--num_worker', default=8)
    parser.add_argument('--iter_sample', default=3)

    # Encoder Traning
    parser.add_argument('--num_layer', default=2)
    parser.add_argument('--num_epoch', default=20)
    parser.add_argument('--interval_save_loss', default=20)
    parser.add_argument('--interval_save_train', default=100)
    parser.add_argument('--interval_save_test', default=2000)
    parser.add_argument('--interval_save_ckpt', default=4000)

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
    parser.add_argument('--use_schedule', default=True)

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
    parser.add_argument('--coef_lpips', type=float, default=0.2)
    parser.add_argument('--coef_adv', type=float, default=0.03)

    # Others
    parser.add_argument('--dim_z', type=int, default=119)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size_batch', default=64)
    parser.add_argument('--port', type=str, default='12355')

    # GPU
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--multi_gpu', default=True)
    parser.add_argument('--amp', default=True)

    return parser.parse_args()


def setup_dist(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train(dev, world_size, config, args,
            dataset=None,
            sample_train=None,
            sample_valid=None,
            path_ckpts=None,
            path_log=None,
        ):
    use_multi_gpu = world_size > 1

    writer = None
    if use_multi_gpu:
        setup_dist(dev, world_size, args.port)
        if dev == 0:
            writer = SummaryWriter(path_log)
    else:
        writer = SummaryWriter(path_log)

    # Setup model
    EG = Colorizer(config, 
                   args.path_ckpt_g, 
                   args.norm_type,
                   id_mid_layer=args.num_layer, 
                   activation=args.activation, 
                   fix_g=(not args.finetune_g),
                   init_e=args.weight_init)
    EG.train()

    # Print Architecture
    if use_multi_gpu:
        if dev == 0:
            print(EG)
    else:
        print(EG)

    D = models.Discriminator(**config)
    D.train()
    if args.use_pretrained_d:
        D.load_state_dict(torch.load(args.path_ckpt_d), strict=False)

    # Load model
    vgg_per = VGG16Perceptual(args.path_vgg).to(dev)
    EG = EG.to(dev)
    D = D.to(dev)
    if use_multi_gpu:
        EG = DDP(EG, device_ids=[dev], 
                 find_unused_parameters=True)
        D = DDP(D, device_ids=[dev], 
                find_unused_parameters=False)
        vgg_per = DDP(vgg_per, device_ids=[dev], 
                      find_unused_parameters=False)

    # Optimizer
    optimizer_g = optim.Adam([p for p in EG.parameters() if p.requires_grad],
            lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d = optim.Adam(D.parameters(),
            lr=args.lr_d, betas=(args.b1_d, args.b2_d))

    # Schedular
    if args.use_schedule:
        scheduler_g = optim.lr_scheduler.LambdaLR(optimizer=optimizer_g,
                                        lr_lambda=lambda epoch: 0.97 ** epoch)
        scheduler_d = optim.lr_scheduler.LambdaLR(optimizer=optimizer_d,
                                        lr_lambda=lambda epoch: 0.97 ** epoch)

    # Datasets
    sampler, sampler_real = None, None
    if use_multi_gpu:
        sampler = DistributedSampler(dataset)

    dataloader = DataLoader(dataset, batch_size=args.size_batch, 
            shuffle=True if sampler is None else False, 
            sampler=sampler, pin_memory=True,
            num_workers=args.num_worker, drop_last=True)

    # AMP
    scaler = None
    if args.amp:
        scaler = GradScaler()

    num_iter = 0
    for epoch in range(args.num_epoch):
        if use_multi_gpu:
            sampler.set_epoch(epoch)
        tbar = tqdm(dataloader)
        tbar.set_description('epoch: %03d' % epoch)
        for i, (x, c) in enumerate(tbar):
            EG.train()

            x, c = x.to(dev), c.to(dev)
            x_gray = transforms.Grayscale()(x)

            # Sample z
            z = torch.zeros((args.size_batch, args.dim_z)).to(dev)
            z.normal_(mean=0, std=0.8)

            # Generate fake image
            if args.amp:
                with autocast():
                    fake = EG(x_gray, c, z)
            else:
                fake = EG(x_gray, c, z)

            # DISCRIMINATOR 
            optimizer_d.zero_grad()
            cal_loss_d = lambda: loss_fn_d(D=D,
                                           c=c,
                                           real=x,
                                           fake=fake.detach())
            if args.amp:
                with autocast():
                    loss_d = cal_loss_d()
                scaler.scale(loss_d).backward()
                scaler.step(optimizer_d)
                scaler.update()
            else:
                loss_d = cal_loss_d() 
                loss_d.backward()
                optimizer_d.step()

            # GENERATOR
            optimizer_g.zero_grad()
            cal_loss_g = lambda: loss_fn_g(D=D,
                                           vgg_per=vgg_per,
                                           x=x,
                                           c=c,
                                           args=args,
                                           fake=fake)
            if args.amp:
                with autocast():
                    loss, loss_dic = cal_loss_g()
                scaler.scale(loss).backward()
                scaler.step(optimizer_g)
                scaler.update()
            else:
                loss, loss_dic = cal_loss_g()
                loss.backward()
                optimizer_g.step()

            loss_dic['loss_d'] = loss_d

            # Logger
            condition = dev == 0 if use_multi_gpu else True
            if num_iter % args.interval_save_loss == 0 and condition:
                make_log_scalar(writer, num_iter, loss_dic)
            if num_iter % args.interval_save_train == 0 and condition:
                make_log_img(EG, args.dim_z, writer, args, sample_train,
                        dev, num_iter, 'train')
            if num_iter % args.interval_save_test == 0 and condition:
                make_log_img(EG, args.dim_z, writer, args, sample_valid,
                        dev, num_iter, 'valid')
            num_iter += 1

        # Save Model
        if use_multi_gpu:
            make_log_ckpt(EG.module, D.module,
                    args, epoch, path_ckpts)
        else:
            make_log_ckpt(EG, D, args, epoch, path_ckpts)

        if args.use_schedule:
            scheduler_d.step(epoch)
            scheduler_g.step(epoch)


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

    if use_multi_gpu:
        args.size_batch = int(args.size_batch / num_gpu)
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
