import os
from os.path import join, exists
import models
from models import Colorizer, VGG16Perceptual

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pickle
import argparse
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from tqdm import tqdm

from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.losses import loss_fn_d, loss_fn_g
from utils.common_utils import (extract_sample, set_seed, prepare_dataset,
                                mk_hint)
from utils.logger import (make_log_img, make_log_ckpt, load_for_retrain,
                          load_for_retrain_EMA)
from utils.common_utils import color_enhacne_blend
import utils

from torch_ema import ExponentialMovingAverage
from functools import partial
import wandb


def args4model(parser):
  parser.add_argument(
      '--norm_type',
      default='adabatch',
      choices=['instance', 'batch', 'layer', 'adain', 'adabatch', 'id'])
  parser.add_argument('--activation',
                      default='relu',
                      choices=['relu', 'lrelu', 'sigmoid'])
  parser.add_argument('--weight_init',
                      default='ortho',
                      choices=['xavier', 'N02', 'ortho', ''])
  parser.add_argument('--dim_z', type=int, default=119)

  parser.add_argument('--dim_f', type=int, default=16)
  parser.add_argument('--no_cond_e', action='store_true')
  parser.add_argument('--chunk_size_z_e', type=int, default=0)
  parser.add_argument('--dim_encoder_c', type=int, default=128)


def args4train(parser):
  parser.add_argument('--retrain', action='store_true')
  parser.add_argument('--retrain_epoch', type=int)
  parser.add_argument('--num_layer', type=int, default=2)
  parser.add_argument('--num_epoch', type=int, default=20)

  parser.add_argument('--finetune_g', default=True)
  parser.add_argument('--finetune_d', default=True)

  parser.add_argument('--num_worker', type=int, default=8)

  # Optimizer
  parser.add_argument("--lr", type=float, default=0.0001)
  parser.add_argument("--b1", type=float, default=0.0)
  parser.add_argument("--b2", type=float, default=0.999)
  parser.add_argument("--lr_d", type=float, default=0.00003)
  parser.add_argument("--b1_d", type=float, default=0.0)
  parser.add_argument("--b2_d", type=float, default=0.999)
  parser.add_argument('--use_schedule', default=True)
  parser.add_argument('--schedule_decay', type=float, default=0.90)
  parser.add_argument('--schedule_type',
                      type=str,
                      default='mult',
                      choices=['mult', 'linear'])

  # Loss
  parser.add_argument('--loss_targets',
                      type=str,
                      nargs='+',
                      required=True,
                      choices=['mse', 'vgg_per', 'adv'])

  parser.add_argument('--coef_mse', type=float, default=1.0)
  parser.add_argument('--coef_vgg_per', type=float, default=0.2)
  parser.add_argument('--coef_zhinge', type=float, default=0.2)
  parser.add_argument('--coef_adv', type=float, default=0.03)
  parser.add_argument('--coef_wip', type=float, default=0.02)
  parser.add_argument('--vgg_target_layers',
                      type=int,
                      nargs='+',
                      default=[1, 2, 13, 20])

  # EMA
  parser.add_argument('--decay_ema_g', type=float, default=0.999)

  # ETC
  parser.add_argument('--use_enhance', action='store_true')
  parser.add_argument('--coef_enhance', type=float, default=1.5)

  parser.add_argument('--std_z', type=float, default=0.8)
  parser.add_argument('--mu_z', type=float, default=1.0)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--size_batch', type=int, default=60)

  parser.add_argument('--index_target',
                      type=int,
                      nargs='+',
                      default=list(range(1000)))
  parser.add_argument('--no_pretrained_g', action='store_true')
  parser.add_argument('--no_pretrained_d', action='store_true')


def args4log(parser):
  parser.add_argument('--interval_save_loss', type=int, default=20)
  parser.add_argument('--interval_save_train', type=int, default=150)
  parser.add_argument('--interval_save_test', type=int, default=200)

  parser.add_argument('--num_test_sample', type=int, default=16)
  parser.add_argument('--num_row_grid', type=int, default=4)


def args4io(parser):
  parser.add_argument('--path_log', default='runs')
  parser.add_argument('--path_ckpts', default='ckpts')
  parser.add_argument('--path_config', default='./pretrained/config.pickle')
  parser.add_argument('--path_vgg', default='./pretrained/vgg16.pickle')
  parser.add_argument('--path_ckpt_g', default='./pretrained/G_ema_256.pth')
  parser.add_argument('--path_ckpt_d', default='./pretrained/D_256.pth')
  parser.add_argument('--path_imgnet_train', default='./imgnet/train')
  parser.add_argument('--path_imgnet_val', default='./imgnet/val')

  parser.add_argument('--no_save', action='store_true')


def parse_args():
  parser = argparse.ArgumentParser()

  args4log(parser)
  args4train(parser)
  args4model(parser)
  args4io(parser)

  # Others
  parser.add_argument('--task_name', default='unknown')
  parser.add_argument('--detail', default='unknown')
  parser.add_argument('--port', type=str, default='12355')

  return parser.parse_args()


def setup_dist(rank, world_size, port):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = port

  # initialize the process group
  dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train(
    dev,
    world_size,
    config,
    args,
    dataset=None,
    sample_train=None,
    sample_valid=None,
    path_ckpts=None,
    path_log=None,
):

  is_main_dev = dev == 0
  setup_dist(dev, world_size, args.port)
  if is_main_dev:
    wandb.login()
    wandb.init(project="my-test-project", config=vars(args))

  # Setup model
  EG = Colorizer(
      config,
      args.path_ckpt_g,
      id_mid_layer=args.num_layer,
      fix_g=(not args.finetune_g),
      load_g=(not args.no_pretrained_g),
      init_w=args.weight_init,
  )
  EG.train()
  D = models.Discriminator(**config)
  D.train()
  if not args.no_pretrained_d:
    D.load_state_dict(torch.load(args.path_ckpt_d, map_location='cpu'),
                      strict=False)

  # Optimizer
  optimizer_g = optim.Adam([p for p in EG.parameters() if p.requires_grad],
                           lr=args.lr,
                           betas=(args.b1, args.b2))
  optimizer_d = optim.Adam(D.parameters(),
                           lr=args.lr_d,
                           betas=(args.b1_d, args.b2_d))

  # Schedular
  if args.use_schedule:
    if args.schedule_type == 'mult':
      schedule = lambda epoch: args.schedule_decay**epoch
    elif args.schedule_type == 'linear':
      schedule = lambda epoch: (args.num_epoch - epoch) / args.num_epoch
    else:
      raise Exception('Invalid shedule type')
    scheduler_g = optim.lr_scheduler.LambdaLR(optimizer=optimizer_g,
                                              lr_lambda=schedule)
    scheduler_d = optim.lr_scheduler.LambdaLR(optimizer=optimizer_d,
                                              lr_lambda=schedule)

  num_iter = 0
  epoch_start = 0
  if args.retrain:
    if args.retrain_epoch is None:
      raise Exception('retrain_epoch is required')
    epoch_start = args.retrain_epoch + 1
    num_iter = load_for_retrain(EG, D, optimizer_g, optimizer_d, scheduler_g,
                                scheduler_d, args.retrain_epoch, path_ckpts,
                                'cpu')
    dist.barrier()

  # Set Device
  EG = EG.to(dev)
  D = D.to(dev)
  vgg_per = VGG16Perceptual(args.path_vgg, args.vgg_target_layers).to(dev)
  utils.optimizer_to(optimizer_g, 'cuda:%d' % dev)
  utils.optimizer_to(optimizer_d, 'cuda:%d' % dev)

  # EMA
  ema_g = ExponentialMovingAverage(EG.parameters(), decay=args.decay_ema_g)
  if args.retrain:
    load_for_retrain_EMA(ema_g, args.retrain_epoch, path_ckpts, 'cpu')

  # DDP
  torch.cuda.set_device(dev)
  torch.cuda.empty_cache()

  EG = DDP(EG, device_ids=[dev], find_unused_parameters=True)
  D = DDP(D, device_ids=[dev], find_unused_parameters=False)
  vgg_per = DDP(vgg_per, device_ids=[dev], find_unused_parameters=True)

  # Datasets
  sampler = DistributedSampler(dataset)
  dataloader = DataLoader(dataset,
                          batch_size=args.size_batch,
                          shuffle=True if sampler is None else False,
                          sampler=sampler,
                          pin_memory=True,
                          num_workers=args.num_worker,
                          drop_last=True)

  color_enhance = partial(color_enhacne_blend, factor=args.coef_enhance)

  # AMP
  scaler = GradScaler()

  loss_dict = {}
  for epoch in range(epoch_start, args.num_epoch):
    sampler.set_epoch(epoch)

    tbar = tqdm(dataloader)
    tbar.set_description('epoch: %03d' % epoch)
    for i, (x_g, x, c) in enumerate(tbar):
      EG.train()

      x_g, x, c = x_g.to(dev), x.to(dev), c.to(dev)

      x_hint = mk_hint(x)
      x_input = torch.cat([x_g, x_hint], dim=-3)

      # Sample z
      z_g = torch.zeros((args.size_batch, args.dim_z)).to(dev)
      z_g.normal_(mean=args.mu_z, std=args.std_z)

      # Generate fake image
      with autocast():
        fake = EG(x_input, c, z_g)

      # DISCRIMINATOR
      x_real = x
      c_real = c

      if args.use_enhance:
        x_real = color_enhance(x)

      optimizer_d.zero_grad()
      with autocast():
        loss_d = loss_fn_d(D=D,
                           c=c_real,
                           real=x_real,
                           fake=fake.detach(),
                           loss_dict=loss_dict)
      scaler.scale(loss_d).backward()
      scaler.step(optimizer_d)
      scaler.update()

      # GENERATOR
      optimizer_g.zero_grad()
      with autocast():
        loss = loss_fn_g(D=D,
                         x=x,
                         c=c,
                         vgg_per=vgg_per,
                         args=args,
                         fake=fake,
                         loss_dict=loss_dict,
                         dev=dev)

      scaler.scale(loss).backward()
      scaler.step(optimizer_g)
      scaler.update()

      # EMA
      if is_main_dev:
        ema_g.update()

      # Logging
      if is_main_dev:
        if num_iter % args.interval_save_loss == 0:

          # We average the accumulated losses for each interval
          for key in loss_dict.keys():
            loss_dict[key] = sum(loss_dict[key]) / len(loss_dict[key])

          # Commit loss data
          wandb.log(loss_dict, step=num_iter)

        if num_iter % args.interval_save_train == 0:
          make_log_img(EG,
                       args.dim_z,
                       args,
                       sample_train,
                       dev,
                       num_iter,
                       'train',
                       ema=ema_g)
        if num_iter % args.interval_save_test == 0:
          make_log_img(EG,
                       args.dim_z,
                       args,
                       sample_valid,
                       dev,
                       num_iter,
                       'valid',
                       ema=ema_g)

      if num_iter % args.interval_save_loss == 0:
        loss_dict.clear()

      num_iter += 1

    # Save Model
    if is_main_dev and not args.no_save:
      make_log_ckpt(EG=EG.module,
                    D=D.module,
                    optim_g=optimizer_g,
                    optim_d=optimizer_d,
                    schedule_g=scheduler_g,
                    schedule_d=scheduler_d,
                    ema_g=ema_g,
                    num_iter=num_iter,
                    args=args,
                    epoch=epoch,
                    path_ckpts=path_ckpts)

    if args.use_schedule:
      scheduler_d.step(epoch)
      scheduler_g.step(epoch)


def main():

  args = parse_args()

  # Note Retrain
  if args.retrain:
    print("This is retrain work after EPOCH %03d" % args.retrain_epoch)

  # GPU OPTIONS
  num_gpu = torch.cuda.device_count()
  if num_gpu == 0:
    raise Exception('No available GPU')
  elif num_gpu == 1:
    print('Use single GPU')
  elif num_gpu > 1:
    print('Use multi GPU: %02d EA' % num_gpu)
  else:
    raise Exception('Invalid GPU setting')

  # Load Configuratuion
  with open(args.path_config, 'rb') as f:
    config = pickle.load(f)

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
  print('logger name:', path_log)

  # DATASETS
  prep = transforms.Compose([
      ToTensor(),
      transforms.Resize(256),
      transforms.CenterCrop(256),
  ])

  dataset, dataset_val = prepare_dataset(args.path_imgnet_train,
                                         args.path_imgnet_val,
                                         args.index_target,
                                         prep=prep)

  is_shuffle = True
  args.size_batch = int(args.size_batch / num_gpu)
  sample_train = extract_sample(dataset,
                                args.num_test_sample,
                                is_shuffle,
                                pin_memory=False)
  sample_valid = extract_sample(dataset_val,
                                args.num_test_sample,
                                is_shuffle,
                                pin_memory=False)

  mp.spawn(train,
           args=(num_gpu, config, args, dataset, sample_train, sample_valid,
                 path_ckpts, path_log),
           nprocs=num_gpu)


if __name__ == '__main__':
 main()
