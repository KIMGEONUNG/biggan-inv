import torch
from os.path import join
from .common_utils import lab_fusion, make_grid_multi
from torch.cuda.amp import autocast


def make_log_ckpt(EG, D, args, num_iter, path_ckpts):

    name = 'D_%03d.ckpt' % num_iter 
    path = join(path_ckpts, name) 
    torch.save(D.state_dict(), path) 

    name = 'EG_%03d.ckpt' % num_iter 
    path = join(path_ckpts, name) 
    torch.save(EG.state_dict(), path) 


def make_log_scalar(writer, num_iter, loss_dic: dict):
    loss_g = loss_dic['loss_g']
    loss_d = loss_dic['loss_d']
    writer.add_scalars('GAN loss', 
        {'G': loss_g.item(), 'D': loss_d.item()}, num_iter)

    del loss_dic['loss_g']
    del loss_dic['loss_d']
    for key, value in loss_dic.items():
        writer.add_scalar(key, value.item(), num_iter)


def make_log_img(EG, dim_z, writer, args, sample, dev, num_iter, name):
    EG.eval()

    outputs_rgb = []
    outputs_fusion = []
    with torch.no_grad():
        for id_sample in range(len(sample['xs'])):
            z = torch.zeros((args.size_batch, dim_z))
            z.normal_(mean=0, std=0.8)
            x_gt = sample['xs'][id_sample]

            x = sample['xs_gray'][id_sample]
            c = sample['cs'][id_sample]
            z, x, c = z.to(dev), x.to(dev), c.to(dev)

            with autocast():
                output = EG(x, c, z)

            output = output.add(1).div(2).detach().cpu()
            output_fusion = lab_fusion(x_gt, output)
            outputs_rgb.append(output)
            outputs_fusion.append(output_fusion)

    grid = make_grid_multi(outputs_rgb, nrow=4)
    writer.add_image('recon_%s_rgb' % name, 
            grid, num_iter)

    grid = make_grid_multi(outputs_fusion, nrow=4)
    writer.add_image('recon_%s_fusion' % name, 
            grid, num_iter)

    writer.flush()
