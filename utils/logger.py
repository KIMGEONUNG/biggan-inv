import torch
from os.path import join
from .common_utils import lab_fusion, mk_hint
from statistics import mean
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import wandb


def make_log_ckpt(EG, D, optim_g, optim_d, schedule_g, schedule_d, ema_g,
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
  torch.save(
      {
          'optim_g': optim_g.state_dict(),
          'optim_d': optim_d.state_dict(),
          'schedule_g': schedule_g.state_dict(),
          'schedule_d': schedule_d.state_dict(),
          'num_iter': num_iter
      }, path)


def load_for_retrain(EG, D, optim_g, optim_d, schedule_g, schedule_d, epoch,
                     path_ckpts, dev):
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


def make_log_img(EG, dim_z, args, sample, dev, num_iter, name, ema=None):
  rgbs = []
  fusions = []
  batch_size = args.size_batch

  xs: torch.Tensor = sample['xs']
  cs: torch.Tensor = sample['cs']
  x_gs: torch.Tensor = sample['x_gs']

  zs = torch.zeros((xs.shape[0], dim_z))
  zs.normal_(mean=args.mu_z, std=args.std_z)

  EG.eval()

  for i in range(len(xs) // batch_size):
    z = zs[batch_size * i:batch_size * (i + 1), ...]
    x = xs[batch_size * i:batch_size * (i + 1), ...]
    c = cs[batch_size * i:batch_size * (i + 1), ...]
    x_g = x_gs[batch_size * i:batch_size * (i + 1), ...]
    z, c, x_g = z.to(dev), c.to(dev), x_g.to(dev)
    x_hint = mk_hint(x).to(dev)
    x_input = torch.cat([x_g, x_hint], dim=-3)

    with torch.no_grad():
      with ema.average_parameters():
        output = EG(x_input, c, z)
      output = output.add(1).div(2).detach().cpu()

    output_fusion = lab_fusion(x, output)

    rgbs.append(output)
    fusions.append(output_fusion)

  rgbs = torch.cat(rgbs, dim=0)
  fusions = torch.cat(fusions, dim=0)

  rgbs = [wandb.Image(ToPILImage()(img)) for img in rgbs]
  fusions = [wandb.Image(ToPILImage()(img)) for img in fusions]
  wandb.log({'%s_rgb' % name: rgbs}, step=num_iter)
  wandb.log({'%s_fusion' % name: fusions}, step=num_iter)
