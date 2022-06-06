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


class ResConvBlock(nn.Module):
    def __init__(self, 
            ch_in, 
            ch_out, 
            ch_c=128, 
            is_down=False,
            dropout=0.2,
            activation='relu', 
            pool='avg', 
            norm='batch', 
            use_res=True, 
            **kwargs):
        super().__init__()

        self.is_down = is_down
        self.has_condition = False
        self.use_res = use_res

        # Convolution
        if self.use_res:
            self.conv = nn.Conv2d(ch_in, ch_out, 
                    kernel_size=1, 
                    stride=1, 
                    padding=0)

        self.conv_1 = nn.Conv2d(ch_in, ch_out, 
                kernel_size=3, 
                stride=1, 
                padding=1)
        self.conv_2 = nn.Conv2d(ch_out, ch_out, 
                kernel_size=3, 
                stride=1, 
                padding=1)

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
            self.dropout=None

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


class EncoderF_Res(nn.Module):

    def __init__(self,
                 dim_in=1,
                 dim_out=768,
                 dim_unit=96,
                 dim_c=128,
                 norm='batch',
                 activation='relu',
                 init='ortho',
                 use_att=False,
                 use_res=True,
                 dropout=[0.2, 0.2, 0.2, 0.2, None],
                 chunk_size_z=0,
                 ):
        super().__init__()

        self.init = init
        self.use_att = use_att

        self.num_blocks = 5

        kwargs = {}
        if activation == 'lrelu':
            kwargs['l_slope'] = 0.2

        if use_att:
            raise NotImplementedError()
            conv4att = functools.partial(
                SNConv2d,
                kernel_size=3,
                padding=1,
                num_svs=1,
                num_itrs=1,
                eps=1e-06)
            self.att = Attention(384, conv4att)

        # output is 96 x 256 x 256
        self.res1 = ResConvBlock(dim_in, dim_unit * 1,
                                 is_down=False, 
                                 activation=activation,
                                 norm=norm,
                                 use_res=use_res,
                                 dropout=dropout[0],
                                 ch_c=dim_c,
                                 **kwargs)
        # output is 192 x 128 x 128 
        self.res2 = ResConvBlock(dim_unit * 1, dim_unit * 2,
                                 is_down=True, 
                                 activation=activation,
                                 norm=norm,
                                 use_res=use_res,
                                 dropout=dropout[1],
                                 ch_c=dim_c,
                                 **kwargs)
        # output is  384 x 64 x 64 
        self.res3 = ResConvBlock(dim_unit * 2, dim_unit * 4,
                                 is_down=True, 
                                 activation=activation,
                                 norm=norm,
                                 use_res=use_res,
                                 dropout=dropout[2],
                                 ch_c=dim_c,
                                 **kwargs)
        # output is  768 x 32 x 32 
        self.res4 = ResConvBlock(dim_unit * 4, dim_unit * 8,
                                 is_down=True, 
                                 activation=activation,
                                 norm=norm,
                                 use_res=use_res,
                                 dropout=dropout[3],
                                 ch_c=dim_c,
                                 **kwargs)
        # output is  768 x 16 x 16 
        self.res5 = ResConvBlock(dim_unit * 8, dim_unit * 8,
                                 is_down=True, 
                                 activation=activation,
                                 norm=norm, 
                                 use_res=use_res,
                                 dropout=dropout[4],
                                 ch_c=dim_c,
                                 **kwargs)
        self.init_weights()

    def forward(self, x, c=None, z=None):
        # # Set condition
        # if self.chunk_size_z != 0:
        #     zs = torch.split(z, self.chunk_size_z, 1)
        #     cs = [torch.cat([c, item], 1) for item in zs]
        # else:
        #     cs = [c] * self.num_blocks

        # Feedforward
        for i in range(0, self.num_blocks):
            block = getattr(self, 'res%d' % (i + 1)) 
            x = block(x, c)
            # x = block(x, cs[i])

        return x

    def init_weights(self):
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    pass
                    # print('Init style not recognized...')


# z: ([batch, 17])
# h: ([batch, 24576])
# index 0 : ([batch, 1536, 4, 4])
# index 1 : ([batch, 1536, 8, 8])
# index 2 : ([batch, 768, 16, 16])
# index 3 : ([batch, 768, 32, 32])
# index 4 : ([batch, 384, 64, 64])
# index 5 : ([batch, 192, 128, 128])
# index 6: ([batch, 96, 256, 256])
# result: ([batch, 3 256, 256])
if __name__ == '__main__':
    pass
