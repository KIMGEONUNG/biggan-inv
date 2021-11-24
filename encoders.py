import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if activation == 'sigmoid':
            blocks.append(nn.Sigmoid())
        elif activation == 'lrelu':
            raise NotImplementedError

        # Dropout
        if dropout is not None:
            blocks.append(nn.Dropout(dropout))

        self.conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.conv_block(x)

# Obsolete for implementation bug
class EncoderF_16(nn.Module):
    """
    input feature: [batch, 3, 256, 256]
    Target feature: [batch, 768, 16, 16]
    """
    def __init__(self, ch_in=1, ch_out=768, ch_unit=96, norm='batch'):
        super().__init__()
        self.net = nn.Sequential(
                ConvBlock(ch_in, ch_unit * 1),
                #  96 x 256 x 256
                ConvBlock(ch_unit * 1, ch_unit * 1, is_down=False,
                    norm=norm),
                ConvBlock(ch_unit * 1, ch_unit * 1, is_down=False,
                    norm=norm),
                ConvBlock(ch_unit * 1, ch_unit * 2, is_down=True,
                    norm=norm),
                #  192 x 128 x 128 
                ConvBlock(ch_unit * 2, ch_unit * 2, is_down=False,
                    norm=norm),
                ConvBlock(ch_unit * 2, ch_unit * 2, is_down=False,
                    norm=norm),
                ConvBlock(ch_unit * 2, ch_unit * 4, is_down=True,
                    norm=norm),
                #  384 x 64 x 64 
                ConvBlock(ch_unit * 4, ch_unit * 4, is_down=False,
                    norm=norm),
                ConvBlock(ch_unit * 4, ch_unit * 4, is_down=False,
                    norm=norm),
                ConvBlock(ch_unit * 4, ch_unit * 8, is_down=True,
                    norm=norm),
                #  768 x 32 x 32 
                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=False,
                    norm=norm),
                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=False,
                    norm=norm),
                ConvBlock(ch_unit * 8, ch_unit * 8, is_down=True,
                    norm=norm),
                #  768 x 16 x 16 

                nn.Conv2d(ch_unit * 8, ch_unit * 8, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1)
                )

    def forward(self, x):
        return self.net(x)


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
    model = EncoderFZ_16()
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(model))
