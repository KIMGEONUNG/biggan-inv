import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class ConvBlock(nn.Module):
    def __init__(self, 
            ch_in, 
            ch_out, 
            is_down=False,
            dropout=0.2,
            norm='batch', 
            activation='relu', 
            **kwargs):
        super().__init__()

        blocks = []
        
        # Convolution
        stride = 1
        if is_down: 
            stride = 2
        conv = nn.Conv2d(ch_in, ch_out, 
                kernel_size=3, 
                stride=stride, 
                padding=1)
        blocks.append(conv)

        # Normalization 
        if norm == 'batch':
            blocks.append(nn.BatchNorm2d(ch_out))
        elif norm == 'instance':
            blocks.append(nn.InstanceNorm2d(ch_out))
        elif norm == 'layer':
            blocks.append(nn.LayerNorm(kwargs['shape']))

        # Nonlinearity 
        if activation == 'relu':
            blocks.append(nn.ReLU(True))
        elif activation == 'sigmoid':
            blocks.append(nn.Sigmoid())
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, True)
        else:
            raise Exception('Invalid nonlinearity')

        # Dropout
        if dropout is not None:
            blocks.append(nn.Dropout(dropout))

        self.conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.conv_block(x)


class AdaIN(nn.Module):

    def __init__(self, dim_ch, dim_cond):
        super().__init__()
        self.dim_ch = dim_ch
        self.dim_cond = dim_cond
        self.cal_m = nn.Linear(dim_cond, dim_ch)
        self.cal_s = nn.Linear(dim_cond, dim_ch)
        self.eps = 1e-05 

    def forward(self, x, c):
        dim_batch = x.shape[0]

        m = self.cal_m(c)
        m = m.view(dim_batch, self.dim_ch, 1, 1)
        s = self.cal_m(c)
        s = s.view(dim_batch, self.dim_ch, 1, 1)

        m_x = x.mean(dim=(-2, -1))
        m_x = m_x.view(dim_batch, self.dim_ch, 1, 1)

        m_s = x.std(dim=(-2, -1))
        m_s = m_s.view(dim_batch, self.dim_ch, 1, 1)

        x = (x - m_x) / (m_s + self.eps) 
        x = x * s + m

        return x


class AdaConvBlock(nn.Module):
    def __init__(self, 
            ch_in, 
            ch_out, 
            ch_c=128, 
            is_down=False,
            dropout=0.2,
            activation='relu', 
            **kwargs):
        super().__init__()

        # Convolution
        stride = 1
        if is_down: 
            stride = 2
        self.conv = nn.Conv2d(ch_in, ch_out, 
                kernel_size=3, 
                stride=stride, 
                padding=1)

        # Normalization 
        self.normalize = AdaIN(ch_out, ch_c)

        # Nonlinearity 
        self.activation = None
        if activation == 'relu':
            self.activation = nn.ReLU(True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, True)
        else:
            raise Exception('Invalid nonlinearity')

        # Dropout
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, c):
        x = self.conv(x)
        x = self.normalize(x, c)
        x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


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
            **kwargs):
        super().__init__()

        self.is_down = is_down
        self.has_condition = False

        # Convolution
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
        if self.is_down:
            x = self.pool(x)
        x = self.conv(x)

        # Merge
        x = x + x_

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class EncoderF_Res(nn.Module):
    """
    input feature: [batch, 3, 256, 256]
    Target feature: [batch, 768, 16, 16]
    """
    def __init__(self, ch_in=1, ch_out=768, ch_unit=96, norm='batch'):
        super().__init__()
        # output is 96 x 256 x 256
        self.res1 = ResConvBlock(ch_in, ch_unit * 1, is_down=False, norm=norm)
        # output is 192 x 128 x 128 
        self.res2 = ResConvBlock(ch_unit * 1, ch_unit * 2, is_down=True, norm=norm)
        # output is  384 x 64 x 64 
        self.res3 = ResConvBlock(ch_unit * 2, ch_unit * 4, is_down=True, norm=norm)
        # output is  768 x 32 x 32 
        self.res4 = ResConvBlock(ch_unit * 4, ch_unit * 8, is_down=True, norm=norm)
        # output is  768 x 16 x 16 
        self.res5 = ResConvBlock(ch_unit * 8, ch_unit * 8, is_down=True, norm=norm)

    def forward(self, x, c=None):
        x = self.res1(x, c)
        x = self.res2(x, c)
        x = self.res3(x, c)
        x = self.res4(x, c)
        x = self.res5(x, c)
        return x


class EncoderF_ada(nn.Module):
    def __init__(self, ch_in=1, ch_out=768, ch_unit=96, ch_c=128, norm='instance'):
        super().__init__()
        self.first = ConvBlock(ch_in, ch_unit * 1, is_down=False,
            norm=norm, shape=(96, 256, 256))

        #  96 x 256 x 256
        self.adaconv1 = AdaConvBlock(ch_unit * 1, ch_unit * 1, ch_c,
            is_down=False, norm=norm, shape=(96, 256, 256))
        self.adaconv2 = AdaConvBlock(ch_unit * 1, ch_unit * 1, ch_c, 
            is_down=False, norm=norm, shape=(96, 256, 256))
        self.adaconv3 = AdaConvBlock(ch_unit * 1, ch_unit * 2, ch_c,
            is_down=True, norm=norm, shape=(192, 128, 128))
        #  192 x 128 x 128 
        self.adaconv4 = AdaConvBlock(ch_unit * 2, ch_unit * 2, ch_c,
            is_down=False, norm=norm, shape=(192, 128, 128))
        self.adaconv5 = AdaConvBlock(ch_unit * 2, ch_unit * 2, ch_c,
            is_down=False, norm=norm, shape=(192, 128, 128))
        self.adaconv6 = AdaConvBlock(ch_unit * 2, ch_unit * 4, ch_c,
            is_down=True, norm=norm, shape=(384, 64, 64))
        #  384 x 64 x 64 
        self.adaconv7 = AdaConvBlock(ch_unit * 4, ch_unit * 4, ch_c,
            is_down=False, norm=norm, shape=(384, 64, 64))
        self.adaconv8 = AdaConvBlock(ch_unit * 4, ch_unit * 4, ch_c,
            is_down=False, norm=norm, shape=(384, 64, 64))
        self.adaconv9 = AdaConvBlock(ch_unit * 4, ch_unit * 8, ch_c,
            is_down=True, norm=norm, shape=(768, 32, 32))
        #  768 x 32 x 32 
        self.adaconv10 = AdaConvBlock(ch_unit * 8, ch_unit * 8, ch_c,
            is_down=False, norm=norm, shape=(768, 32, 32))
        self.adaconv11 = AdaConvBlock(ch_unit * 8, ch_unit * 8, ch_c,
            is_down=False, norm=norm, shape=(768, 32, 32))
        self.adaconv12 = AdaConvBlock(ch_unit * 8, ch_unit * 8, ch_c,
            is_down=True, norm=norm, shape=(768, 16, 16))
        #  768 x 16 x 16 

        self.last = nn.Conv2d(ch_unit * 8, ch_unit * 8, 
            kernel_size=3, 
            stride=1, 
            padding=1)

    def forward(self, x, c):
        x = self.first(x)
        x = self.adaconv1(x, c)
        x = self.adaconv2(x, c)
        x = self.adaconv3(x, c)
        x = self.adaconv4(x, c)
        x = self.adaconv5(x, c)
        x = self.adaconv6(x, c)
        x = self.adaconv7(x, c)
        x = self.adaconv8(x, c)
        x = self.adaconv9(x, c)
        x = self.adaconv10(x, c)
        x = self.adaconv11(x, c)
        x = self.adaconv12(x, c)
        x = self.last(x)
        return x


class EncoderZ(nn.Module):
    """
    input feature: [batch, 3, 256, 256]
    Target feature: [batch, 768, 16, 16]
    """
    def __init__(self, ch_in=1, ch_out=119, ch_unit=96, norm='batch'):
        super().__init__()
        self.cnn = nn.Sequential(
                ConvBlock(ch_in, ch_unit * 1, is_down=False,
                    norm=norm, shape=(96, 256, 256)),
                #  96 x 256 x 256

                ConvBlock(ch_unit * 1, ch_unit * 2, is_down=True,
                    norm=norm, shape=(192, 128, 128)),
                #  192 x 128 x 128 

                ConvBlock(ch_unit * 2, ch_unit * 4, is_down=True,
                    norm=norm, shape=(384, 64, 64)),
                #  384 x 64 x 64 

                ConvBlock(ch_unit * 4, ch_unit * 8, is_down=True,
                    norm=norm, shape=(768, 32, 32)),
                #  768 x 32 x 32 

                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=True,
                    norm=norm, shape=(768, 16, 16)),
                #  768 x 16 x 16 

                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=True,
                    norm=norm, shape=(768, 8, 8)),
                #  768 x 16 x 16 

                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=True,
                    norm=norm, shape=(768, 4, 4)),
                #  768 x 16 x 16 

                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=True,
                    norm=norm, shape=(768, 2, 2)),
                #  768 x 16 x 16 

                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=True,
                    norm=norm, shape=(768, 1, 1)),
                #  768 x 16 x 16 
                )
        self.mlp = nn.Linear(ch_unit * 8, ch_out)

    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x.view(x.size()[0], -1))
        return x


class EncoderF_16(nn.Module):
    """
    input feature: [batch, 3, 256, 256]
    Target feature: [batch, 768, 16, 16]
    """
    def __init__(self, ch_in=1, ch_out=768, ch_unit=96, norm='batch'):
        super().__init__()
        self.net = nn.Sequential(
                ConvBlock(ch_in, ch_unit * 1, is_down=False,
                    norm=norm, shape=(96, 256, 256)),
                #  96 x 256 x 256
                ConvBlock(ch_unit * 1, ch_unit * 1, is_down=False,
                    norm=norm, shape=(96, 256, 256)),
                ConvBlock(ch_unit * 1, ch_unit * 1, is_down=False,
                    norm=norm, shape=(96, 256, 256)),
                ConvBlock(ch_unit * 1, ch_unit * 2, is_down=True,
                    norm=norm, shape=(192, 128, 128)),
                #  192 x 128 x 128 
                ConvBlock(ch_unit * 2, ch_unit * 2, is_down=False,
                    norm=norm, shape=(192, 128, 128)),
                ConvBlock(ch_unit * 2, ch_unit * 2, is_down=False,
                    norm=norm, shape=(192, 128, 128)),
                ConvBlock(ch_unit * 2, ch_unit * 4, is_down=True,
                    norm=norm, shape=(384, 64, 64)),
                #  384 x 64 x 64 
                ConvBlock(ch_unit * 4, ch_unit * 4, is_down=False,
                    norm=norm, shape=(384, 64, 64)),
                ConvBlock(ch_unit * 4, ch_unit * 4, is_down=False,
                    norm=norm, shape=(384, 64, 64)),
                ConvBlock(ch_unit * 4, ch_unit * 8, is_down=True,
                    norm=norm, shape=(768, 32, 32)),
                #  768 x 32 x 32 
                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=False,
                    norm=norm, shape=(768, 32, 32)),
                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=False,
                    norm=norm, shape=(768, 32, 32)),
                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=True,
                    norm=norm, shape=(768, 16, 16)),
                #  768 x 16 x 16 

                nn.Conv2d(ch_unit * 8, ch_unit * 8, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1)
                )

    def forward(self, x):
        return self.net(x)


class EncoderFZ_16_Multi(nn.Module):
    """
    Target feature: [batch, 768, 16, 16]
    """
    def __init__(self, ch_in=1, ch_out=768, ch_unit=96, ch_z=119):
        super().__init__()
        self.net1 = nn.Sequential(
                ConvBlock(ch_in, ch_unit * 1),
                #  96 x 256 x 256

                ConvBlock(ch_unit * 1, ch_unit * 1, is_down=False),
                ConvBlock(ch_unit * 1, ch_unit * 1, is_down=False),
                ConvBlock(ch_unit * 1, ch_unit * 2, is_down=True),
                #  192 x 128 x 128 
                nn.Conv2d(ch_unit * 2, ch_unit * 2, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1)
                )

        self.net2 = nn.Sequential(
                ConvBlock(ch_unit * 2, ch_unit * 2, is_down=False),
                ConvBlock(ch_unit * 2, ch_unit * 2, is_down=False),
                ConvBlock(ch_unit * 2, ch_unit * 4, is_down=True),
                #  384 x 64 x 64 
                nn.Conv2d(ch_unit * 4, ch_unit * 4, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1)
                )

        self.net3 = nn.Sequential(
                ConvBlock(ch_unit * 4, ch_unit * 4, is_down=False),
                ConvBlock(ch_unit * 4, ch_unit * 4, is_down=False),
                ConvBlock(ch_unit * 4, ch_unit * 8, is_down=True),
                #  768 x 32 x 32 
                nn.Conv2d(ch_unit * 8, ch_unit * 8, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1)
                )

        self.net4 = nn.Sequential(
                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=False),
                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=False),
                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=True),
                #  768 x 16 x 16 
                nn.Conv2d(ch_unit * 8, ch_unit * 8, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1)
                )
                #  768 x 16 x 16 

        self.net5 = nn.Sequential(
                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=False),
                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=False),
                ConvBlock(ch_unit * 8, ch_unit * 16, is_down=True),
                # 1536 x 8 x 8 

                ConvBlock(ch_unit * 16, ch_unit * 16, is_down=True),
                #  1536 x 4 x 4 
                ConvBlock(ch_unit * 16, ch_unit * 16, is_down=True),
                #  1536 x 2 x 2 
                ConvBlock(ch_unit * 16, ch_unit * 16, is_down=True),
                #  1536 x 1 x 1 
                ConvBlock(ch_unit * 16, ch_z, is_down=False),
                )

    def forward(self, x):
        f1 = self.net1(x)
        f2 = self.net2(f1)
        f3 = self.net3(f2)
        f4 = self.net4(f3)
        z = self.net5(f4)
        z = z.view(z.shape[0], -1)
        return [f1, f2, f3, f4], z 


class EncoderFZ_16(nn.Module):
    """
    Target feature: [batch, 768, 16, 16]
    """
    def __init__(self, ch_in=1, ch_out=768, ch_unit=96, ch_z=119):
        super().__init__()
        self.net1 = nn.Sequential(
                ConvBlock(ch_in, ch_unit * 1),
                #  96 x 256 x 256

                ConvBlock(ch_unit * 1, ch_unit * 1, is_down=False),
                ConvBlock(ch_unit * 1, ch_unit * 1, is_down=False),
                ConvBlock(ch_unit * 1, ch_unit * 2, is_down=True),
                #  192 x 128 x 128 
                
                ConvBlock(ch_unit * 2, ch_unit * 2, is_down=False),
                ConvBlock(ch_unit * 2, ch_unit * 2, is_down=False),
                ConvBlock(ch_unit * 2, ch_unit * 4, is_down=True),
                #  384 x 64 x 64 

                ConvBlock(ch_unit * 4, ch_unit * 4, is_down=False),
                ConvBlock(ch_unit * 4, ch_unit * 4, is_down=False),
                ConvBlock(ch_unit * 4, ch_unit * 8, is_down=True),
                #  768 x 32 x 32 

                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=False),
                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=False),
                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=True),
                #  768 x 16 x 16 

                nn.Conv2d(ch_unit * 8, ch_unit * 8, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1)
                )
                #  768 x 16 x 16 

        self.net2 = nn.Sequential(
                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=False),
                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=False),
                ConvBlock(ch_unit * 8, ch_unit * 16, is_down=True),
                # 1536 x 8 x 8 

                ConvBlock(ch_unit * 16, ch_unit * 16, is_down=True),
                #  1536 x 4 x 4 
                ConvBlock(ch_unit * 16, ch_unit * 16, is_down=True),
                #  1536 x 2 x 2 
                ConvBlock(ch_unit * 16, ch_unit * 16, is_down=True),
                #  1536 x 1 x 1 
                ConvBlock(ch_unit * 16, ch_z, is_down=False),
                )

    def forward(self, x):
        f = self.net1(x)
        z = self.net2(f)
        z = z.view(z.shape[0], -1)
        return f, z 
    

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
    # model = EncoderF_Res()
    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(count_parameters(model))
    from models.layers import GBlock
    model = GBlock(6, 7)
    print(model)


    pass
