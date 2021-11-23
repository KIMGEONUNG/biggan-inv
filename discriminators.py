import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class Discriminator_F(nn.Module):
    def __init__(self):
        super().__init__()

        def block_d(ch_in, ch_out, down=False):
            block = [
                    nn.Conv2d(ch_in, ch_out, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                    ]
            if down:
                block += [
                    nn.Conv2d(ch_out, ch_out, 3, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                    ]

            return block

        self.model = nn.Sequential(
            *block_d(1024, 512),  # 16
            *block_d(512, 256, True), # 16
            *block_d(256, 128), # 8 
            *block_d(128, 64, True),  # 4 
        )

        # The height and width of downsampled image
        ds_size = 4 
        self.adv_layer = nn.Sequential(nn.Linear(64 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class D_Down(nn.Module):
    def __init__(self, size_target=5):
        """
        size_target : this is power of 2
        """
        super().__init__()

        def gen_block(in_filters, out_filters, bn=True):
            block = [
                    nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                    ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.resize = transforms.Resize(size_target)
        models = [] 
        models += gen_block(3, 512)

        for _ in range(size_target - 1):
            models += gen_block(512, 512)

        self.model = nn.Sequential(*models)
        # The height and width of downsampled image
        self.adv_layer = nn.Sequential(
                nn.Linear(512, 1),
                nn.Sigmoid())

    def forward(self, x):
        x = self.resize(x)
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        validity = self.adv_layer(x)

        return validity


class DCGAN_D(nn.Module):
    def __init__(self):
        super().__init__()

        def gen_block(in_filters, out_filters, bn=True):
            block = [
                    nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                    ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *gen_block(3, 16, bn=False), # 128 x 128
            *gen_block(16, 32), # 64 x 64
            *gen_block(32, 64), # 32 x 32
            *gen_block(64, 128), # 16 x 16
            *gen_block(128, 256), # 8 x 8 
            *gen_block(256, 512), # 4 x 4 
        )

        # The height and width of downsampled image
        ds_size = 4 
        self.adv_layer = nn.Sequential(
                nn.Linear(512 * ds_size ** 2, 1),
                nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
