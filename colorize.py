import os
from skimage import color
import numpy as np
from torch.utils.data import DataLoader
import models
from encoders import EncoderF_16, EncoderF_ada
from train import Colorizer
import torch
import pickle
import argparse
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
from train import prepare_dataset, extract_sample


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2)
    # 5 --> 32, 4 --> 16, ...
    parser.add_argument('--max_iter', default=1000)
    parser.add_argument('--num_row', type=int, default=8)
    parser.add_argument('--class_index', type=int, default=15)
    parser.add_argument('--size_batch', type=int, default=8)
    parser.add_argument('--num_worker', default=8)

    # I/O
    parser.add_argument('--path_config', default='config.pickle')
    parser.add_argument('--path_ckpt_eg', default='./ckpts/mark/EG_0042600.ckpt')
    parser.add_argument('--path_args', default='./ckpts/mark//args.pkl')
    parser.add_argument('--path_output', default='./out_colorize')
    parser.add_argument('--path_imgnet_train', default='./imgnet/train')
    parser.add_argument('--path_imgnet_val', default='./imgnet/val')

    parser.add_argument('--num_layer', default=2)
    parser.add_argument('--norm_type', default='instance', 
            choices=['instance', 'batch', 'layer'])

    # Dataset
    parser.add_argument('--iter_sample', default=4)
    parser.add_argument('--dim_z', type=int, default=119)

    # User Input 
    parser.add_argument('--index_target',
            type=int, nargs='+', default=list(range(20)))
            # type=int, nargs='+', default=[0, 1 ,2 ,3 ,4, 11, 14, 15])
    parser.add_argument('--index_force',
            type=int, default=None)
            # type=int, default=14)
    parser.add_argument('--color_jitter', type=int, default=1)
    parser.add_argument('--z_sample_scheme', type=str, 
            default='sample', choices=['sample', 'zero', 'one'])
    parser.add_argument('--colorization_target', default='valid',
            choices=['valid', 'train']
            )

    # Conversion
    parser.add_argument('--device', default='cuda:1')

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


def fusion(x_l, x_ab):
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


def main(args):
    print(args)

    if args.seed >= 0:
        set_seed(args.seed)

    dev = args.device

    prep = transforms.Compose([
            ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            ])
    dataset, dataset_val = prepare_dataset(
            args.path_imgnet_train,
            args.path_imgnet_val,
            args.index_target,
            prep=prep)
    if args.colorization_target  == 'train':
        print('Load train dataset')
        dataset_target = dataset 
    elif args.colorization_target  == 'valid':
        print('Load valid dataset')
        dataset_target = dataset_val
    else:
        raise Exception('Invalid colorization target')

    dataloader = DataLoader(dataset_target, batch_size=args.size_batch,
            shuffle=True, num_workers=args.num_worker, drop_last=True)

    # Load Configuratuion
    with open(args.path_config, 'rb') as f:
        config = pickle.load(f)
    with open(args.path_args, 'rb') as f:
        args_loaded = pickle.load(f)

    # Load Model 
    EG = Colorizer(config, args.path_ckpt_eg, args.norm_type,
            id_mid_layer=args.num_layer)
    EG.load_state_dict(torch.load(args.path_ckpt_eg), strict=True)
    EG.eval()
    EG.to(dev)

    if not os.path.exists(args.path_output):
        os.mkdir(args.path_output)

    with torch.no_grad():
        num_iter = 0
        for i, (x, c) in enumerate(tqdm(dataloader)):
            for _ in range(args.color_jitter):
                x, c = x.to(dev), c.to(dev)
                x_gray = transforms.Grayscale()(x)

                # Sample z
                if args.z_sample_scheme == 'sample':
                    z = torch.zeros((args.size_batch, args.dim_z)).to(dev)
                    z.normal_(mean=0, std=0.8)
                elif args.z_sample_scheme == 'zero': 
                    z = torch.zeros((args.size_batch, args.dim_z)).to(dev)
                elif args.z_sample_scheme == 'one': 
                    z = torch.ones((args.size_batch, args.dim_z)).to(dev)
                else:
                    raise Exception('Invalid z sample scheme')

                # Force Index
                if args.index_force is not None:
                    c = ((c / c) * args.index_force).long()

                output = EG(x_gray, c, z)
                output = output.add(1).div(2)

                # LAB
                labs = fusion(x.detach().cpu(), output.detach().cpu())

                # Save Result
                grid_gt = make_grid(x, nrow=args.num_row)
                grid_gray = make_grid(x_gray, nrow=args.num_row)
                grid_out = make_grid(output, nrow=args.num_row)
                grid_lab = make_grid(labs, nrow=args.num_row).to(dev)
                grid = torch.cat([grid_gt, grid_gray, grid_out, grid_lab], dim=-2)
                im = ToPILImage()(grid)
                im.save('./%s/%03d.jpg' % (args.path_output, num_iter))

                num_iter += 1
                if num_iter > args.max_iter:
                    break

if __name__ == '__main__':
    args = parse()
    main(args)
