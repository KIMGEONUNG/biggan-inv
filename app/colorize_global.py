import sys
import os
sys.path.append(os.path.abspath(os.curdir))
from PIL import Image
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
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import lpips
from kornia.color.lab import rgb_to_lab


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2)
    # 5 --> 32, 4 --> 16, ...
    parser.add_argument('--max_iter', default=1000)
    parser.add_argument('--num_row', type=int, default=8)
    parser.add_argument('--size_batch', type=int, default=8)
    parser.add_argument('--num_worker', default=8)
    parser.add_argument('--epoch', type=int, default=12)

    # I/O
    parser.add_argument('--path_vgg', default='./pretrained/vgg16.pickle')
    parser.add_argument('--path_config', default='./pretrained/config.pickle')
    parser.add_argument('--path_ckpt_g', default='./pretrained/G_ema_256.pth')
    parser.add_argument('--path_ckpt', default='./ckpts/baseline_1000')
    parser.add_argument('--path_output', default='./exprs/global')
    parser.add_argument('--path_input', default='./resource/grays_nature/48794_970.jpg')
    parser.add_argument('--path_input_refer', default='./resource/refers/sunset.jpg')
    parser.add_argument('--use_ema', action='store_true')

    parser.add_argument('--num_layer', default=2)
    parser.add_argument('--postfix', default='')

    # Dataset
    parser.add_argument('--dim_z', type=int, default=119)

    # User Input 
    parser.add_argument('--use_rgb', action='store_true')
    parser.add_argument('--max_img', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--class_id', type=int, default=15)
    parser.add_argument('--num_iter', type=int, default=200)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--lpips_net', type=str, default='alex',
                                    choices=['vgg', 'alex'])
    parser.add_argument('--loss', type=str, default='lpips',
                                    choices=['mse', 'lpips', 'feat_vgg'])
    parser.add_argument('--optimizer', type=str, default='adam',
                                    choices=['adam', 'sgd'])
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


class lpips_loss(object):

    def __init__(self, net, dev, use_norm):
        self.model = lpips.LPIPS(net=net).to(dev)
        self.use_norm = use_norm
    
    def __call__(self, x1, x2):
        if self.use_norm:
            x1 = x1.mul(2).sub(-1)
            x2 = x2.mul(2).sub(-1)
         
        return self.model(x1, x2)


class VGGFeatLoss(nn.Module):

    def __init__(self, 
                 path_vgg: str, 
                 resize=True, 
                 normalized_input=True):
        super().__init__()

        import pickle 
        with open(path_vgg, 'rb') as f:
            self.model = pickle.load(f).eval()

        self.normalized_intput = normalized_input
        # self.idx_targets = [1, 2, 13, 20]
        self.idx_targets = [4, 5, 6, 7]

        preprocess = []
        if resize:
            preprocess.append(transforms.Resize((224, 224)))
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

    def forward(self, x1, x2):
        size_batch = x1.shape[0]
        x1_feats = self.get_mid_feats(x1)
        x2_feats = self.get_mid_feats(x2)

        loss = 0
        for feat1, feat2 in zip(x1_feats, x2_feats):
            loss += feat1.sub(feat2).pow(2).mean()

        return loss / size_batch


def get_ab_dist(x_rgb):

    x_lab = rgb_to_lab(x_rgb)

    L, a, b = torch.split(x_lab, 1)
    a, b = a.reshape(-1), b.reshape(-1)

    a_dist = torch.histogram(a, bins=22, range=(-110, 110)) 
    a_dist /= len(a)
    b_dist = torch.histogram(b, bins=22, range=(-110, 110)) 
    b_dist /= len(b)

    return a_dist, b_dist


def main(args):
    size_target = 256

    if args.seed >= 0:
        set_seed(args.seed)

    print('Target checkpoint is %s' % args.path_ckpt)
    print('Target Epoch is %03d' % args.epoch)
    print('Target classes is', args.class_id)
    print('Target loss is', args.loss)

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

    prep=transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale()])


    gray = Image.open(args.path_input) 
    gray = prep(gray)

    refer = Image.open(args.path_input_refer)
    refer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((250, 250))])(refer)

    EG = Colorizer(config, args.path_ckpt_g, args_loaded.norm_type,
            id_mid_layer=args.num_layer)
    EG.load_state_dict(torch.load(path_eg, map_location='cpu'), strict=True)
    EG_ema = ExponentialMovingAverage(EG.parameters(), decay=0.99)
    EG_ema.load_state_dict(torch.load(path_eg_ema, map_location='cpu'))

    EG.eval()
    EG.float()
    EG.to(dev)

    if args.loss == 'lpips':
        loss_fn = lpips_loss(args.lpips_net, dev, True)
    elif args.loss == 'mse':
        loss_fn = nn.MSELoss()
    elif args.loss == 'feat_vgg':
        loss_fn = VGGFeatLoss(args.path_vgg).to(dev)
    else:
        raise Exception()
    
    if args.use_ema:
        print('Use EMA')
        EG_ema.copy_to()

    if not os.path.exists(args.path_output):
        os.mkdir(args.path_output)

    x = gray.clone()
    size = x.shape[1:]

    x = x.to(dev)
    x = x.unsqueeze(0)
    x_resize = transforms.Resize((size_target))(x)

    z = torch.zeros((1, args.dim_z)).to(dev)
    z.normal_(mean=0, std=0.8)

    refer = refer.to(dev)
    refer_dist = get_ab_dist(refer)
    print(refer_dist)
    exit()


    c = torch.LongTensor([args.class_id]).to(dev)
    c_embd = EG.G.shared(c).clone().detach()
    c_embds = [Variable(c_embd.clone(), requires_grad=True) 
               for _ in range(2)]
    

    if args.optimizer == 'adam':
        optimizer = optim.Adam
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD
    else:
        raise Exception("Invalid optimizer")


    optimizer = optimizer(params=c_embds, lr=args.lr)

    with torch.no_grad():
        output = EG.forward_with_c2(x_resize, c_embds, z)
        output_init = output.add(1).div(2).detach()

    tbar = tqdm(range(args.num_iter))
    for j in tbar:

        output = EG.forward_with_c2(x_resize, c_embds, z)
        output = output.add(1).div(2)

        output_fore = output.clone()
        output_back = output.clone()
        m_resize = transforms.Resize(output.shape[-2:])(m)
        x_resize_gt = transforms.Resize(output.shape[-2:])(x)

        output_fore[m_resize == 0] = 0

        output_back[m_resize != 0] = 0
        output_init[m_resize != 0] = 0

        loss = loss_fn(output_fore, m_resize) +\
                loss_fn(output_back, output_init)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tbar.set_postfix(loss=loss.item())

        if j % args.interval == 0:
            x_ = x.clone()
            x_ = x_.squeeze(0).cpu()
            output = output.squeeze(0)
            output = output.detach().cpu()
            output = transforms.Resize(size)(output)

            if args.use_rgb:
                pass
            else:
                output = fusion(x_, output)
            im = ToPILImage()(output)
            im.save('./%s/n%i_iter_%04d%s.jpg' % (args.path_output, i, j, args.postfix))


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
