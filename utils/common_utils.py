import argparse
import logging

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import dataset

from .biggan_utils import CenterCropLongEdge



def save_img(image, path):
    image = np.uint8(255 * (image.cpu().detach().numpy() + 1) / 2.)
    image = np.transpose(image, (1, 2, 0))
    image = Image.fromarray(image)
    image.save(path)


def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.

    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()

    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")

    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(
        tensor.narrow(int(dim), int(start), int(length))
        for start, length in zip(splits, split_sizes))


class LRScheduler(object):

    def __init__(self, optimizer, warm_up):
        super(LRScheduler, self).__init__()
        self.optimizer = optimizer
        self.warm_up = warm_up

    def update(self, iteration, learning_rate, num_group=1000, ratio=1):
        if iteration < self.warm_up:
            learning_rate *= iteration / self.warm_up
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i >= num_group:
                param_group['lr'] = 0
            else:
                param_group['lr'] = learning_rate * ratio**i


def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l


def map_func(storage, location):
    return storage.cuda()


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]
