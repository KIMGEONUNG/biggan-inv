import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .encoders import EncoderF_Res
from .biggan import Generator

class VGG16Perceptual(nn.Module):

    def __init__(self, path_vgg: str, resize=True, normalized_input=True):
        super().__init__()

        import pickle 
        with open(path_vgg, 'rb') as f:
            self.model = pickle.load(f).eval()

        self.normalized_intput = normalized_input
        self.idx_targets = [1, 2, 13, 20]

        preprocess = []
        if resize:
            preprocess.append(transforms.Resize(256))
            preprocess.append(transforms.CenterCrop(224))
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
        x1_feats = self.preprocess(x1)
        x2_feats = self.preprocess(x2)

        loss = 0
        for feat1, feat2 in zip(x1_feats, x2_feats):
            loss += feat1.sub(feat2).pow(2).mean()

        return loss / size_batch


class Colorizer(nn.Module):
    def __init__(self, 
                 config, 
                 path_ckpt_g, 
                 norm_type,
                 activation='relu',
                 id_mid_layer=2, 
                 fix_g=False,
                 init_e=None,
                 use_attention=False):
        super().__init__()

        self.id_mid_layer = id_mid_layer  
        self.use_attention = use_attention

        self.E = EncoderF_Res(norm=norm_type,
                              activation=activation,
                              init=init_e,
                              use_att=use_attention)

        self.G = Generator(**config)
        self.G.load_state_dict(torch.load(path_ckpt_g, map_location='cpu'),
                               strict=False)
        self.fix_g = fix_g
        if fix_g:
            for p in self.G.parameters():
                p.requires_grad = False

    def forward(self, x_gray, c, z):
        f = self.E(x_gray, self.G.shared(c)) 
        output = self.G.forward_from(z, self.G.shared(c), 
                self.id_mid_layer, f)
        return output

    def train(self, mode=True):
        if self.fix_g:
            self.E.train(mode)
        else:
            super().train(mode)

