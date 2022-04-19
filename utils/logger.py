import torch
import torch.nn as nn
from os.path import join
from .common_utils import lab_fusion, make_grid_multi
from torch.cuda.amp import autocast
from statistics import mean
from torchvision.utils import make_grid
from representation import RGBuvHistBlock


def make_log_ckpt(EG, D,
                  optim_g, optim_d,
                  schedule_g, schedule_d, 
                  ema_g, 
                  num_iter, args, epoch, path_ckpts):
    # Encoder&Generator
    name = 'EG_%03d.ckpt' % epoch 
    path = join(path_ckpts, name) 
    torch.save(EG.state_dict(), path) 

    # Discriminator
    name = 'D_%03d.ckpt' % epoch 
    path = join(path_ckpts, name) 
    torch.save(D.state_dict(), path) 

    # EMA Encoder&Generator
    name = 'EG_EMA_%03d.ckpt' % epoch 
    path = join(path_ckpts, name) 
    torch.save(ema_g.state_dict(), path) 

    # Oters
    name = 'OTHER_%03d.ckpt' % epoch 
    path = join(path_ckpts, name) 
    torch.save({'optim_g': optim_g.state_dict(),
                'optim_d': optim_d.state_dict(),
                'schedule_g': schedule_g.state_dict(),
                'schedule_d': schedule_d.state_dict(),
                'num_iter': num_iter}, path)


def load_for_retrain(EG, D,
                     optim_g, optim_d, schedule_g, schedule_d, 
                     epoch, path_ckpts, dev):
    # Encoder&Generator
    name = 'EG_%03d.ckpt' % epoch 
    path = join(path_ckpts, name) 
    state = torch.load(path, map_location=dev)
    EG.load_state_dict(state)

    # Discriminator
    name = 'D_%03d.ckpt' % epoch 
    path = join(path_ckpts, name) 
    state = torch.load(path, map_location=dev)
    D.load_state_dict(state)

    # Oters
    name = 'OTHER_%03d.ckpt' % epoch 
    path = join(path_ckpts, name) 
    state = torch.load(path, map_location=dev)
    optim_g.load_state_dict(state['optim_g'])
    optim_d.load_state_dict(state['optim_d'])
    schedule_g.load_state_dict(state['schedule_g'])
    schedule_d.load_state_dict(state['schedule_d'])

    return state['num_iter']

def load_for_retrain_EMA(ema_g, epoch, path_ckpts, dev):
    # Encoder&Generator
    name = 'EG_EMA_%03d.ckpt' % epoch 
    path = join(path_ckpts, name) 
    state = torch.load(path, map_location=dev)
    ema_g.load_state_dict(state)


def make_log_scalar(writer, num_iter, loss_dict: dict, num_sample):
    loss_g = mean(loss_dict['adv_g'])
    loss_d = mean(loss_dict['adv_d'])
    writer.add_scalars('GAN', 
        {'adv_g': loss_g, 'adv_d': loss_d}, num_iter)

    del loss_dict['adv_g']
    del loss_dict['adv_d']
    for key, value in loss_dict.items():
        writer.add_scalar(key, mean(value), num_iter)
    loss_dict.clear()


def make_log_img(EG, dim_z, writer, args, sample, dev, num_iter, name,
        ema=None):
    outputs_rgb = []
    outputs_fusion = []
    batch_size = args.size_batch * args.num_copy

    xs: torch.Tensor = sample['xs']
    cs: torch.Tensor = sample['cs']
    x_gs: torch.Tensor = sample['x_gs']

    xs = xs.repeat_interleave(args.num_copy_test, dim=0)
    cs = cs.repeat_interleave(args.num_copy_test, dim=0)
    x_gs = x_gs.repeat_interleave(args.num_copy_test, dim=0)
    zs = torch.zeros((xs.shape[0], dim_z))
    zs.normal_(mean=args.mu_z, std=args.std_z)
    
    EG.eval()

    for i in range(len(xs) // batch_size):
        z = zs[batch_size * i: batch_size * (i + 1), ...]
        x = xs[batch_size * i: batch_size * (i + 1), ...]
        c = cs[batch_size * i: batch_size * (i + 1), ...]
        x_g = x_gs[batch_size * i: batch_size * (i + 1), ...]
        z, c, x_g = z.to(dev), c.to(dev), x_g.to(dev)

        with torch.no_grad():
            if ema is None:
                output = EG(x_g, c, z)
            else:
                with ema.average_parameters():
                    output = EG(x_g, c, z)
            output = output.add(1).div(2).detach().cpu()

        output_fusion = lab_fusion(x, output)

        outputs_rgb.append(output)
        outputs_fusion.append(output_fusion)

    outputs_rgb = torch.cat(outputs_rgb, dim=0)
    outputs_fusion = torch.cat(outputs_fusion, dim=0)

    grid = make_grid(outputs_rgb, nrow=4)
    writer.add_image('recon_%s_rgb' % name, 
            grid, num_iter)
    grid = make_grid(outputs_fusion, nrow=4)
    writer.add_image('recon_%s_fusion' % name, 
            grid, num_iter)

    if 'color_scatter_score' in args.eval_targets:
        num_sample = args.num_copy_test
        mse = nn.MSELoss()
        with torch.no_grad():
            feats = RGBuvHistBlock(device=dev)(outputs_rgb.to(dev)).to(dev)
            score = 0 
            num_cal = 0
            for feat in feats.split(num_sample):
                for i in range(num_sample - 1):
                    for j in range(i, num_sample):
                        score += mse(feats[i], feats[j])
                        num_cal += 1
            score /= num_cal
        writer.add_scalar('color_scatter_score', score, num_iter)

    writer.flush()
