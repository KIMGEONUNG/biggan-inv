import os
from os.path import join, exists
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from train import Colorizer
import torch
import pickle
import argparse
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2)
    # 5 --> 32, 4 --> 16, ...
    parser.add_argument('--max_iter', default=1000)
    parser.add_argument('--num_row', type=int, default=8)
    parser.add_argument('--class_index', type=int, default=15)
    parser.add_argument('--size_batch', type=int, default=8)
    parser.add_argument('--num_worker', default=8)
    parser.add_argument('--epoch', type=int, default=0)

    # I/O
    parser.add_argument('--path_config', default='./pretrained/config.pickle')
    parser.add_argument('--path_ckpt_g', default='./pretrained/G_ema_256.pth')
    parser.add_argument('--path_ckpt', default='./ckpts/baseline_1000')
    parser.add_argument('--path_output', default='./results')
    parser.add_argument('--path_imgnet_train', default='./imgnet/train')
    parser.add_argument('--path_imgnet_val', default='./imgnet/val')
    parser.add_argument('--use_ema', action='store_true')

    parser.add_argument('--num_layer', default=2)
    parser.add_argument('--norm_type', default='instance', 
            choices=['instance', 'batch', 'layer'])

    # Dataset
    parser.add_argument('--iter_sample', default=4)
    parser.add_argument('--dim_z', type=int, default=119)

    # User Input 
    parser.add_argument('--index_target',
            type=int, nargs='+', default=list(range(1000)))
    parser.add_argument('--color_jitter', type=int, default=1)
    parser.add_argument('--z_sample_scheme', type=str, 
            default='sample', choices=['sample', 'zero', 'one'])
    parser.add_argument('--colorization_target', default='valid',
            choices=['valid', 'train']
            )

    parser.add_argument('--raw_save', action='store_true')
    parser.add_argument('--view_gt', default=True)
    parser.add_argument('--view_gray', default=True)
    parser.add_argument('--view_rgb', default=False)
    parser.add_argument('--view_lab', default=True)

    parser.add_argument('--device', default='cuda:0')

    return parser.parse_args()


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


def main(args):
    size_target = 256

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

    grays = ImageFolder(args.path_imgnet_val,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Resize(size_target),
                            # transforms.CenterCrop(size_target),
                            transforms.Grayscale()]))

    EG = Colorizer(config, args.path_ckpt_g, args_loaded.norm_type,
            id_mid_layer=args.num_layer)
    EG.load_state_dict(torch.load(path_eg, map_location='cpu'), strict=True)
    EG_ema = ExponentialMovingAverage(EG.parameters(), decay=0.99)
    EG_ema.load_state_dict(torch.load(path_eg_ema, map_location='cpu'))

    EG.eval()
    EG.float()
    EG.to(dev)
    
    if args.use_ema:
        print('Use EMA')
        EG_ema.copy_to()

    if not os.path.exists(args.path_output):
        os.mkdir(args.path_output)

    for i, (x, c) in enumerate(tqdm(grays)):
        size = x.shape[1:]

        c = torch.LongTensor([c])
        x = x.unsqueeze(0)
        x, c = x.to(dev), c.to(dev)
        z = torch.zeros((1, args.dim_z)).to(dev)
        z.normal_(mean=0, std=0.8)

        x_resize = transforms.Resize((size_target))(x)
        with torch.no_grad():
            output = EG(x_resize, c, z)
            output = output.add(1).div(2)

        x = x.squeeze(0).cpu()
        output = output.squeeze(0)
        output = output.detach().cpu()
        output = transforms.Resize(size)(output)

        lab = fusion(x, output)
        im = ToPILImage()(lab)
        im.save('./%s/%05d.jpg' % (args.path_output, i))


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
