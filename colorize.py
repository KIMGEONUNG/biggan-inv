import os
from skimage import color
import numpy as np
from torch.utils.data import DataLoader
import models
from encoders import EncoderF_16
import torch
import pickle
import argparse
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2)
    # 5 --> 32, 4 --> 16, ...
    parser.add_argument('--max_iter', default=100)
    parser.add_argument('--num_layer', default=2)
    parser.add_argument('--num_row', type=int, default=8)
    parser.add_argument('--class_index', type=int, default=15)
    parser.add_argument('--size_batch', type=int, default=8)

    # I/O
    parser.add_argument('--path_config', default='config.pickle')
    parser.add_argument('--path_ckpt_g', default='./ckpts/encoder_f_16_finetune_v1/G_0016200.ckpt')
    parser.add_argument('--path_ckpt_e', default='./ckpts/encoder_f_16_finetune_v1/E_0016200.ckpt')
    parser.add_argument('--path_output', default='./out_colorize')

    parser.add_argument('--colorization_target', default='valid',
            choices=['valid', 'train']
            )
    parser.add_argument('--path_dataset_encoder_train', default='./dataset_encoder/')
    parser.add_argument('--path_dataset_encoder_valid', default='./dataset_encoder_val/')

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

    # Load Configuratuion
    with open(args.path_config, 'rb') as f:
        config = pickle.load(f)

    # Load Generator
    G = models.Generator(**config)
    G.load_state_dict(torch.load(args.path_ckpt_g), strict=False)
    G.to(dev)
    G.eval()

    # Load Encoder
    encoder = EncoderF_16(norm='instance').to(dev)
    encoder.load_state_dict(torch.load(args.path_ckpt_e))
    encoder.eval()

    # Latents
    c = torch.ones(args.size_batch) * args.class_index
    c = c.to(dev).long()

    # Datasets
    prep = transforms.Compose([
                ToTensor(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
            ])

    path_root = None
    if args.colorization_target  == 'train':
        path_root = args.path_dataset_encoder_train
    elif args.colorization_target  == 'valid':
        path_root = args.path_dataset_encoder_valid

    dataset = ImageFolder(path_root, transform=prep)
    dataloader  = DataLoader(dataset, batch_size=args.size_batch, shuffle=False,
            num_workers=8, drop_last=True)
    
    if not os.path.exists(args.path_output):
        os.mkdir(args.path_output)

    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(dataloader)):
            x_gt = x.clone().to(dev)
            grid_gt = make_grid(x_gt, nrow=args.num_row)

            # Sample z
            z = torch.zeros((args.size_batch, G.dim_z)).to(dev)
            z.normal_(mean=0, std=0.8)

            x = x.to(dev)
            x = transforms.Grayscale()(x)
            grid_gray = make_grid(x, nrow=args.num_row)

            f = encoder(x) # [batch, 1024, 16, 16]
            output = G.forward_from(z, G.shared(c), args.num_layer, f)
            output = output.add(1).div(2)
            grid_out = make_grid(output, nrow=args.num_row)

            # LAB
            labs = fusion(x_gt.detach().cpu(), output.detach().cpu())
            grid_lab = make_grid(labs, nrow=args.num_row).to(dev)

            grid = torch.cat([grid_gt, grid_gray, grid_out, grid_lab], dim=-2)
            im = ToPILImage()(grid)

            im.save('./%s/%03d.jpg' % (args.path_output, i))

            if i > args.max_iter:
                break

if __name__ == '__main__':
    args = parse()
    main(args)
