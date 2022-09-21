import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from .layers import SNConv2d, Attention
from torch.nn import init


class ClassConditionNorm(nn.Module):

  def __init__(self,
               output_size,
               input_size,
               which_linear=functools.partial(nn.Linear, bias=False),
               eps=1e-5,
               norm_style='bn'):
    super().__init__()
    self.output_size, self.input_size = output_size, input_size
    # Prepare gain and bias layers
    self.gain = which_linear(input_size, output_size)
    self.bias = which_linear(input_size, output_size)
    # epsilon to avoid dividing by 0
    self.eps = eps
    self.norm_style = norm_style

    self.register_buffer('stored_mean', torch.zeros(output_size))
    self.register_buffer('stored_var', torch.ones(output_size))

  def forward(self, x, y):
    # Calculate class-conditional gains and biases
    gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
    bias = self.bias(y).view(y.size(0), -1, 1, 1)

    if self.norm_style == 'bn':
      out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                         self.training, 0.1, self.eps)
    elif self.norm_style == 'in':
      out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None,
                            self.training, 0.1, self.eps)

    return out * gain + bias


class ConvBlock(nn.Module):

  def __init__(self,
               ch_in,
               ch_out,
               ch_c=128,
               is_down=False,
               dropout=0.2,
               activation='relu',
               pool='avg',
               norm='batch',
               use_res=False,
               **kwargs):
    super().__init__()

    self.is_down = is_down
    self.has_condition = False
    self.use_res = use_res

    # Convolution
    if self.use_res:
      self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    self.conv_1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
    self.conv_2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)

    # Normalization
    if norm == 'batch':
      self.normalize_1 = nn.BatchNorm2d(ch_in)
      self.normalize_2 = nn.BatchNorm2d(ch_out)
    elif norm == 'id':
      self.normalize_1 = nn.Identity()
      self.normalize_2 = nn.Identity()
    elif norm == 'instance':
      self.normalize_1 = nn.InstanceNorm2d(ch_in)
      self.normalize_2 = nn.InstanceNorm2d(ch_out)
    elif norm == 'layer':
      self.normalize_1 = nn.LayerNorm(kwargs['l_norm_shape_1'])
      self.normalize_2 = nn.LayerNorm(kwargs['l_norm_shape_2'])
    elif norm == 'adain':
      self.has_condition = True
      self.normalize_1 = ClassConditionNorm(ch_in, ch_c, norm_style='in')
      self.normalize_2 = ClassConditionNorm(ch_out, ch_c, norm_style='in')
    elif norm == 'adabatch':
      self.has_condition = True
      self.normalize_1 = ClassConditionNorm(ch_in, ch_c, norm_style='bn')
      self.normalize_2 = ClassConditionNorm(ch_out, ch_c, norm_style='bn')
    else:
      raise Exception('Invalid Normalization')

    # Nonlinearity
    self.activation = None
    if activation == 'relu':
      self.activation = lambda x: F.relu(x, True)
    elif activation == 'sigmoid':
      self.activation = F.sigmoid
    elif activation == 'lrelu':
      slope = kwargs['l_slope']
      self.activation = lambda x: F.leaky_relu(x, slope, True)
    else:
      raise Exception('Invalid Nonlinearity')

    # Pooling
    self.pool = None
    if pool == 'avg':
      self.pool = lambda x: F.avg_pool2d(x, kernel_size=2)
    elif pool == 'max':
      self.pool = lambda x: F.max_pool2d(x, kernel_size=2)
    elif pool == 'min':
      self.pool = lambda x: F.min_pool2d(x, kernel_size=2)
    else:
      raise Exception('Invalid Pooling')

    # Dropout
    if dropout is not None:
      self.dropout = nn.Dropout(dropout)
    else:
      self.dropout = None

  def forward(self, x, c=None):

    # Residual Path
    x_ = x

    if self.has_condition:
      x_ = self.normalize_1(x_, c)
    else:
      x_ = self.normalize_1(x_)
    x_ = self.activation(x_)

    if self.is_down:
      x_ = self.pool(x_)

    x_ = self.conv_1(x_)

    if self.has_condition:
      x_ = self.normalize_2(x_, c)
    else:
      x_ = self.normalize_2(x_)
    x_ = self.activation(x_)
    x_ = self.conv_2(x_)

    # Main Path
    if self.use_res:
      if self.is_down:
        x = self.pool(x)
      x = self.conv(x)
    else:
      x = 0

    # Merge
    x = x + x_

    if self.dropout is not None:
      x = self.dropout(x)

    return x


class EncoderF(nn.Module):

  def __init__(
      self,
      dim_in=[4, 96 * 1, 96 * 2, 96 * 4, 96 * 8],
      dim_out=[96 * 1, 96 * 2, 96 * 4, 96 * 8, 96 * 8],
      dim_c=[128, 128, 128, 128, 128],
      num_block=5,
      norm=['adabatch', 'adabatch', 'adabatch', 'adabatch', 'adabatch'],
      activation=['relu', 'relu', 'relu', 'relu', 'relu'],
      downsample=[False, True, True, True, True],
      conditions=[False, True, True, True, True],
      respath=[True, True, True, True, True],
      dropout=[None, None, None, None, None],
      init_w='ortho',
  ):
    super().__init__()

    self.init_w = init_w

    kwargs = {}
    if activation == 'lrelu':
      kwargs['l_slope'] = 0.2

    blocks = []

    for i in range(num_block):
      blocks.append(
          ConvBlock(dim_in[i],
                    dim_out[i],
                    is_down=downsample[i],
                    activation=activation[i],
                    norm=norm[i],
                    use_res=respath[i],
                    dropout=dropout[i],
                    ch_c=dim_c[i],
                    **kwargs))

    self.blocks = nn.Sequential(*blocks)

    self.init_weights()

  def forward(self, x, c=None, z=None):

    for block in self.blocks:
      x = block(x, c)

    return x

  def init_weights(self):
    for module in self.modules():
      if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
          or isinstance(module, nn.Embedding)):
        if self.init_w == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init_w == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init_w in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          pass
          # print('Init style not recognized...')


if __name__ == '__main__':
  pass
