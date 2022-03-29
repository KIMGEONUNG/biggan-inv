import torch
import torch.nn as nn
from torch.utils.data import Subset 
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Grayscale
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from skimage import color
from .color_models import rgb2lab, lab2rgb
from typing import Tuple, Any, Callable, Optional
from PIL import Image
import pickle


LAYER_DIM = {
        0: [1536, 4],
        1: [1536, 8],
        2: [768, 16],
        3: [768, 32],
        4: [384, 64],
        4: [192, 128],
        }


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def set_seed(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_inf_batch(loader):
    while True:
        for x in loader:
            yield x


def copy_buff(m_from: nn.Module, m_to: nn.Module):
    for (k1, v1), (k2, v2) in zip(m_from.named_buffers(), m_to.named_buffers()):
        assert k1 == k2
        v2.copy_(v1)


def extract(dataset, target_ids):
    '''
    extract data element based on class index
    '''
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in target_ids:
            indices.append(i)
    return Subset(dataset, indices)


def prepare_dataset(
        path_train,
        path_valid,
        index_target,
        prep=transforms.Compose([
            ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
        ])
):

    dataset = SemanticDataset(path_train, 'train', 'semantic_map_train', transform=prep)
    dataset = extract(dataset, index_target)

    dataset_val = SemanticDataset(path_valid, 'valid', 'semantic_map_valid', transform=prep)
    dataset_val = extract(dataset_val, index_target)
    return dataset, dataset_val


def extract_sample(dataset, size_batch, num_iter, is_shuffle):
    dataloader = DataLoader(dataset, batch_size=size_batch,
            shuffle=is_shuffle, num_workers=4, pin_memory=True,
            drop_last=True)
    xs = []
    xgs = []
    cs = []
    for i, (x, c, smt) in enumerate(dataloader):
        if i >= num_iter:
            break
        xg = transforms.Grayscale()(x)
        xs.append(x), cs.append(c), xgs.append(xg)
    return {'xs': xs, 'cs': cs, 'xs_gray': xgs}


def make_grid_multi(xs, nrow=4):
    return make_grid(torch.cat(xs, dim=0), nrow=nrow)


def lab_fusion(img_gray, img_rgb):
    img_gray *= 100
    ab = rgb2lab(img_rgb)[..., 1:, :, :]
    lab = torch.cat([img_gray, ab], dim=1)
    rgb = lab2rgb(lab)
    return rgb 


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def label_to_one_hot_label(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.

    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([
                [[0, 1], 
                [2, 0]]
            ])
        >>> label_to_one_hot_label(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """

    shape = labels.shape
    # one hot : (B, C=num_classes, H, W)
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)
    
    # labels : (B, H, W)
    # labels.unsqueeze(1) : (B, C=1, H, W)
    # ret : (B, C=num_classes, H, W)
    ret = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) # + eps    
    return ret


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.JPEG', '.png', '.ppm', '.bmp')


class SemanticDataset(DatasetFolder):

    def __init__(
            self,
            root: str,
            name_from: str,
            name_to: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(root=root,
                         loader=loader,
                         extensions=IMG_EXTENSIONS if is_valid_file is None
                         else None,
                         transform=transform,
                         target_transform=target_transform,
                         is_valid_file=is_valid_file)

        self.imgs = self.samples
        self.name_from = name_from
        self.name_to = name_to

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        path_img, cls = self.samples[index]
        path_smt = path_img.replace(self.name_from, self.name_to)\
                           .replace('JPEG', 'dat')
        
        sample = self.loader(path_img)
        with open(path_smt, 'rb') as f:
            semantic_map = pickle.load(f)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            cls = self.target_transform(cls)

        return sample, cls, semantic_map
