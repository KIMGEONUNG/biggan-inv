import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import rgb2lab
from torchvision.transforms import Resize
from representation import RGBuvHistBlock


def enroll_loss(name, loss, loss_dict):
    keys = loss_dict.keys()
    loss_dict[name] = [loss] if name not in keys else loss_dict[name] + [loss]


def loss_fn_d(D, c, real, fake, loss_dict):
    real = (real - 0.5) * 2
    critic_real, _ = D(real, c)
    critic_fake, _ = D(fake, c)
    d_loss_real, d_loss_fake = loss_hinge_dis(critic_fake, critic_real)
    loss_d = (d_loss_real + d_loss_fake) / 2  
    
    enroll_loss('adv_d', loss_d.item(), loss_dict)

    return loss_d


def loss_fn_g(D, x, c, args, fake, vgg_per, loss_dict, dev=None):
    loss = 0
    if 'adv' in args.loss_targets:
        critic, _ = D(fake, c)
        loss_g = loss_hinge_gen(critic) * args.coef_adv
        loss += loss_g 
        enroll_loss('adv_g', loss_g.item(), loss_dict)

    fake_ranged = fake.add(1).div(2)
    if 'mse' in args.loss_targets:
        loss_mse = args.coef_mse * nn.MSELoss()(x, fake_ranged)
        loss += loss_mse
        enroll_loss('mse', loss_mse.item(), loss_dict)

    if 'vgg_per' in args.loss_targets:
        loss_vgg_per = args.coef_vgg_per * vgg_per(x, fake_ranged)
        loss += loss_vgg_per
        enroll_loss('vgg_per', loss_vgg_per.item(), loss_dict)

    return loss


class JSD(nn.Module):

    def __init__(self):
        super(JSD, self).__init__()

    def forward(self, prop_1, prop_2):
        m = 0.5 * (prop_1 + prop_1)
        loss = F.kl_div(prop_1, m, reduce=True)\
             + F.kl_div(prop_2, m, reduce=True)

        return (0.5 * loss)


class color_histogram_loss(nn.Module):

    def __init__(self,
                 resize: int = None,
                 num_bin: int = 220,
                 min: int = -110,
                 max: int = 110,
                 color_space: str = 'ab'
                 ):
        super().__init__()

        if resize is None:
            self.resize = resize
        else:
            self.resize = Resize(resize)

        self.num_bin = num_bin
        self.min = min
        self.max = max
        self.color_space = color_space
        self.dist_fn = JSD()

    def get_histo(self, x):
        # Color distribution is robust with respect to resize operation.
        # So, this resizing trick improves computational efficiency
        if self.resize is not None:
            x = self.resize(x)

        # Extract 'a' and 'b' color space from RGB
        # a, b = rgb2lab(x)[1, :, :], rgb2lab(x)[2, :, :]
        if self.color_space == 'ab':
            chs = rgb2lab(x)[1, :, :], rgb2lab(x)[2, :, :]
        if self.color_space == 'rgb':
            chs = x[0, :, :], x[1, :, :], x[2, :, :]

        # Convert a color data into a color distribution
        hists = []
        for ch in chs:
            hist = torch.histc(ch, self.num_bin, min=self.min, max=self.max)
            hist /= hist.sum()
            hists.append(hist)

        return hists

    def histo_loss(self, x1, x2):
        x1_hists = self.get_histo(x1)
        x2_hists = self.get_histo(x2)

        loss = 0
        for x1_hist, x2_hist in zip(x1_hists, x2_hists):
            loss += self.dist_fn(x1_hist, x2_hist)

        return loss

    def histo_loss_avg(self, xs):
        length = len(xs)

        losses = []
        for i in range(length - 1):
            for j in range(i, length):
                loss = self.histo_loss(xs[i], xs[j])
                losses.append(loss.view(1))

        losses = torch.concat(losses)
        loss_avg = losses.mean()

        return loss_avg

    def forward(self, xs, num_copy):
        xs_group = xs.split(num_copy)

        loss = 0
        for xs in xs_group:
            loss += self.histo_loss_avg(xs)
        loss /= num_copy
        return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


class PerceptLoss(object):

    def __init__(self):
        pass

    def __call__(self, LossNet, fake_img, real_img):
        with torch.no_grad():
            real_feature = LossNet(real_img.detach())
        fake_feature = LossNet(fake_img)
        perceptual_penalty = F.mse_loss(fake_feature, real_feature)
        return perceptual_penalty

    def set_ftr_num(self, ftr_num):
        pass


class DiscriminatorLoss(object):

    def __init__(self, ftr_num=4, data_parallel=False):
        self.data_parallel = data_parallel
        self.ftr_num = ftr_num

    def __call__(self, D, fake_img, real_img):
        if self.data_parallel:
            with torch.no_grad():
                d, real_feature = nn.parallel.data_parallel(
                    D, real_img.detach())
            d, fake_feature = nn.parallel.data_parallel(D, fake_img)
        else:
            with torch.no_grad():
                d, real_feature = D(real_img.detach())
            d, fake_feature = D(fake_img)
        D_penalty = 0
        for i in range(self.ftr_num):
            f_id = -i - 1
            D_penalty = D_penalty + F.l1_loss(fake_feature[f_id],
                                              real_feature[f_id])
        return D_penalty

    def set_ftr_num(self, ftr_num):
        self.ftr_num = ftr_num
