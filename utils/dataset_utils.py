from torchvision.datasets import ImageFolder
from typing import Tuple, Any, Optional, Callable
from torchvision.transforms import Grayscale


class GrayGTPairDataset(ImageFolder):

    def __init__(self,
                 root,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform)

        self.togray = Grayscale()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, label = self.samples[index]
        x_gt = self.loader(path)
        if self.transform is not None:
            x_gt = self.transform(x_gt)
        if self.target_transform is not None:
            label = self.target_transform(label)

        x_gray = self.togray(x_gt)

        return x_gray, x_gt, label
