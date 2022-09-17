import torch
import torch.nn as nn
from torch.utils.data import Subset
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Grayscale, Resize
from math import ceil
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import numpy as np
from skimage import color
from .color_models import rgb2lab, lab2rgb
from .dataset_utils import GrayGTPairDataset
from pycomar.datasets import ImageNetIndexDataset
from random import randint

LAYER_DIM = {
    0: [1536, 4],
    1: [1536, 8],
    2: [768, 16],
    3: [768, 32],
    4: [384, 64],
    4: [192, 128],
}


def optimizer_to(optim, device):
  for param in optim.state.values():
    # Not sure there are any global tensors in the state dict
    if isinstance(param, torch.Tensor):
      param.data = param.data.to(device)
      if param._grad is not None:
        param._grad.data = param._grad.data.to(device)
    elif isinstance(param, dict):
      for subparam in param.values():
        if isinstance(subparam, torch.Tensor):
          subparam.data = subparam.data.to(device)
          if subparam._grad is not None:
            subparam._grad.data = subparam._grad.data.to(device)


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


def copy_buff(m_from: nn.Module, m_to: nn.Module):
  for (k1, v1), (k2, v2) in zip(m_from.named_buffers(), m_to.named_buffers()):
    assert k1 == k2
    v2.copy_(v1)


def filter_dataset(dataset, target_ids):
  '''
    extract data element based on class index
    '''
  indices = []
  for i in range(len(dataset.targets)):
    if dataset.targets[i] in target_ids:
      indices.append(i)
  return Subset(dataset, indices)


def prepare_dataset(path_train,
                    path_valid,
                    index_target,
                    prep_train,
                    prep_valid=ToTensor()):

  dataset = ImageNetIndexDataset(path_train,
                                 transform=prep_train,
                                 post_processings=[Grayscale()])
  dataset = filter_dataset(dataset, index_target)

  dataset_val = ImageNetIndexDataset(path_valid,
                                     transform=prep_valid,
                                     post_processings=[Grayscale()])
  dataset_val = filter_dataset(dataset_val, index_target)
  return dataset, dataset_val


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


def color_enhacne_blend(x, factor):
  x_g = Grayscale(3)(x)
  out = x_g * (1.0 - factor) + x * factor
  out[out < 0] = 0
  out[out > 1] = 1
  return out


def color_enhacne_abgc(x, factor):
  lab = rgb2lab(x)

  ab = lab[..., 1:3, :, :]
  ab /= 110

  ab[ab > 0] = torch.pow(factor, 1 / ab[ab > 0])
  ab[ab < 0] = -torch.pow(factor, 1 / ab[ab < 0].abs())

  ab *= 110
  lab[..., 1:3, :, :] = ab
  rgb = lab2rgb(lab)

  return rgb


def make_grid_multi(xs, nrow=4):
  return make_grid(torch.cat(xs, dim=0), nrow=nrow)


def mk_hint(x: torch.Tensor,
            size_patch=5,
            num_patch=20,
            colorspace='lab',
            use_mask=True):
  hint = torch.zeros_like(x)
  mask = torch.zeros_like(x)[..., :1, :, :]
  h, w = x.shape[-2:]

  for i in range(num_patch):
    coord_h = randint(0, h - size_patch)
    coord_w = randint(0, w - size_patch)

    patch = x[..., coord_h:coord_h + size_patch, coord_w:coord_w + size_patch]
    patch = patch.mean(dim=(-1, -2), keepdim=True)

    hint[..., coord_h:coord_h + size_patch,
         coord_w:coord_w + size_patch] = patch

    mask[..., coord_h:coord_h + size_patch, coord_w:coord_w + size_patch] = 1.0

  if colorspace == 'lab':
    hint = rgb2lab(hint)[..., 1:, :, :] / 110

  if use_mask:
    hint = torch.cat([hint, mask], dim=-3)

  return hint
 

def resizer3unit(x, unit):
  width = x.shape[-2]
  hight = x.shape[-1]

  unit_w = ceil(width / unit)
  unit_h = ceil(hight / unit)

  width_n = unit_w * unit
  hight_n = unit_h * unit
  x_hat = Resize((width_n, hight_n))(x)

  return x_hat
