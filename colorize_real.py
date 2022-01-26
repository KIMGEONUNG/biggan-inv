import os
from os import listdir
from os.path import join, exists
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from train import Colorizer
import torch
import pickle
import argparse
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage, Grayscale, Resize
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
from utils.common_utils import set_seed
from PIL import Image
import timm


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2)

    # I/O
    parser.add_argument('--path_config', default='./pretrained/config.pickle')
    parser.add_argument('--path_ckpt_g', default='./pretrained/G_ema_256.pth')
    parser.add_argument('--path_ckpt', default='./ckpts/baseline_1000')
    parser.add_argument('--path_output', default='./results_real')
    parser.add_argument('--path_input', default='./resource/real_grays')

    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--no_resize', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epoch', type=int, default=0)

    # Setting
    parser.add_argument('--size_target', type=int, default=256)
    parser.add_argument('--cls_model', type=str, default='tf_efficientnet_l2_ns_475')
    # parser.add_argument('--cls_model', type=str, default='resnet50d')

    return parser.parse_args()

MODEL2SIZE = {'resnet50d': 224,
              'tf_efficientnet_l2_ns_475': 475}


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

    # Load Colorizer
    EG = Colorizer(config, 
                   args.path_ckpt_g,
                   args_loaded.norm_type,
                   id_mid_layer=args_loaded.num_layer,
                   activation=args_loaded.activation, 
                   use_attention=args_loaded.use_attention)
    EG.load_state_dict(torch.load(path_eg, map_location='cpu'), strict=True)
    EG_ema = ExponentialMovingAverage(EG.parameters(), decay=0.99)
    EG_ema.load_state_dict(torch.load(path_eg_ema, map_location='cpu'))

    EG.eval()
    EG.float()
    EG.to(dev)
    
    if args.use_ema:
        print('Use EMA')
        EG_ema.copy_to()

    # Load Classifier
    classifier = timm.create_model(
            args.cls_model,
            pretrained=True,
            num_classes=1000
            ).to(dev)
    classifier.eval()
    size_cls = MODEL2SIZE[args.cls_model]

    if not os.path.exists(args.path_output):
        os.mkdir(args.path_output)

    paths = [join(args.path_input, p) for p in listdir(args.path_input)]
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

        x_cls = x.repeat(1,3,1,1)
        x_cls = Resize((size_cls,size_cls))(x_cls)
        c = classifier(x_cls)
        c = torch.argmax(c, dim=-1)

        x_resize = transforms.Resize((args.size_target))(x)
        with torch.no_grad():
            output = EG(x_resize, c, z)
            output = output.add(1).div(2)

        x = x.squeeze(0).cpu()
        x_resize = x_resize.squeeze(0).cpu()
        output = output.squeeze(0)
        output = output.detach().cpu()

        output = transforms.Resize(size)(output)
        lab_fusion = fusion(x, output)
        im = ToPILImage()(lab_fusion)

        name = path.split('/')[-1]
        path_out = join(args.path_output, name)
        im.save(path_out)


def fusion(gray, color):
    # Resize
    light = gray.permute(1, 2, 0).numpy() * 100

    color = color.permute(1, 2, 0)
    color = rgb2lab(color)
    ab = color[:, :, 1:]

    lab = np.concatenate((light, ab), axis=-1)
    lab = lab2rgb(lab)
    lab = torch.from_numpy(lab)
    lab = lab.permute(2, 0, 1)
     
    return lab


if __name__ == '__main__':
    args = parse()
    main(args)
