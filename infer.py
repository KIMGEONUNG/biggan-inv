import os
from os import listdir
from os.path import join, exists
# from skimage.color import rgb2lab, lab2rgb
import glob
import numpy as np
from train import Colorizer
import torch
import torch.nn as nn
import pickle
import argparse
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage, Grayscale, Resize, Compose
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
from utils.common_utils import set_seed, rgb2lab, lab2rgb
from pycomar.datasets import IMAGENET_INDEX
from pycomar.images.colorspace import fuse_luma_chroma
from PIL import Image
import timm
from math import ceil

MODEL2SIZE = {'resnet50d': 224, 'tf_efficientnet_l2_ns_475': 475}


def parse():
  parser = argparse.ArgumentParser()

  parser.add_argument('--seed', type=int, default=2)

  # I/O
  parser.add_argument('--path_config', default='./pretrained/config.pickle')
  parser.add_argument('--path_ckpt_g', default='./pretrained/G_ema_256.pth')
  parser.add_argument('--path_ckpt', default='./ckpts/birds.patch')
  parser.add_argument('--path_output', default='./results_real')
  parser.add_argument('--path_input', default='./samples')

  parser.add_argument('--use_ema', action='store_true')
  parser.add_argument('--use_rgb', action='store_true')
  parser.add_argument('--no_upsample', action='store_true')
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--epoch', type=int, default=19)

  # Setting
  parser.add_argument(
      '--type_resize',
      type=str,
      default='powerof',
      choices=['absolute', 'original', 'square', 'patch', 'powerof'])
  parser.add_argument('--num_power', type=int, default=4)
  parser.add_argument('--size_target', type=int, default=256)
  parser.add_argument('--topk', type=int, default=1)
  parser.add_argument('--cls_model',
                      type=str,
                      default='tf_efficientnet_l2_ns_475')

  return parser.parse_args()


def main(args):

  if args.seed >= 0:
    set_seed(args.seed)

  print('Target Epoch is %03d' % args.epoch)

  path_eg = join(args.path_ckpt, 'EG_%03d.ckpt' % args.epoch)
  path_eg_ema = join(args.path_ckpt, 'EG_EMA_%03d.ckpt' % args.epoch)
  path_args = join(args.path_ckpt, 'args.pkl')

  if not exists(path_eg):
    raise FileNotFoundError(path_eg)
  if not exists(path_args):
    raise FileNotFoundError(path_args)

  # Load Configuratuion
  with open(args.path_config, 'rb') as f:
    config = pickle.load(f)
  with open(path_args, 'rb') as f:
    args_loaded = pickle.load(f)

  dev = args.device

  EG: nn.Module = Colorizer(
      config,
      args.path_ckpt_g,
  )
  EG.load_state_dict(torch.load(path_eg, map_location='cpu'), strict=True)
  EG_ema = ExponentialMovingAverage(EG.parameters(), decay=0.99)
  EG_ema.load_state_dict(torch.load(path_eg_ema, map_location='cpu'))

  EG.eval()
  EG.float()
  EG.to(dev)
  EG_ema.copy_to()

  if not os.path.exists(args.path_output):
    os.mkdir(args.path_output)

  paths = [f for f in glob.glob(args.path_input + '/*/*', recursive=True)]

  resizer = None
  if args.type_resize == 'absolute':
    resizer = Resize((args.size_target))
  elif args.type_resize == 'original':
    resizer = Compose([])
  elif args.type_resize == 'square':
    resizer = Resize((args.size_target, args.size_target))
  elif args.type_resize == 'powerof':

    def resizer(x):
      width = x.shape[-2]
      hight = x.shape[-1]

      unit_w = ceil(width / (2 ** args.num_power))
      unit_h = ceil(hight / (2 ** args.num_power))

      width_n = unit_w * (2 ** args.num_power)
      hight_n = unit_h * (2 ** args.num_power)
      fn = Resize((width_n, hight_n))

      return fn(x)

  elif args.type_resize == 'patch':
    resizer = Resize((args.size_target))
  else:
    raise Exception('Invalid resize type')

  for path in tqdm(paths):
    im = Image.open(path)
    x = ToTensor()(im)
    if x.shape[0] != 1:
      x = Grayscale()(x)

    size = x.shape[1:]

    x = x.unsqueeze(0)
    x = x.to(dev)
    z = torch.zeros((1, args_loaded.dim_z)).to(dev)
    z.normal_(mean=0, std=0.8)

    # Classification
    cls = path.split('/')[-2]
    c = IMAGENET_INDEX[cls]
    c = torch.LongTensor([c]).to(dev)

    x_resize = resizer(x)

    with torch.no_grad():
      output = EG(x_resize, c, z)
      output = output.add(1).div(2)

    if args.no_upsample:
      size_output = x_resize.shape[-2:]
      x_rs = x_resize.squeeze(0).cpu()
    else:
      size_output = size
      x_rs = x.squeeze(0).cpu()

    output = transforms.Resize(size_output)(output)
    output = output.squeeze(0)
    output = output.detach().cpu()

    if args.use_rgb:
      x_img = output
    else:
      x_img = fuse_luma_chroma(x_rs, output)
    im = ToPILImage()(x_img)

    name = path.split('/')[-1].split('.')[0]
    name = name + '_c%03d.jpg' % c.item()

    path_out = join(args.path_output, name)
    im.save(path_out)


if __name__ == '__main__':
  args = parse()
  main(args)
