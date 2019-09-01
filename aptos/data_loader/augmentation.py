import numpy as np
import torch
import torchvision.transforms as T

from .preprocess import ImgProcessor


class AugmentationBase:

    def __init__(self, train):
        self.train = train
        self.transform = self.build_transforms()

    def build_transforms(self):
        raise NotImplementedError('Not implemented!')

    def __call__(self, images):
        return self.transform(images)


class InplacePngTransforms(AugmentationBase):

    def __init__(self, train, img_size):
        self.img_size = img_size
        self.processor = ImgProcessor()
        super().__init__(train)

    def build_transforms(self):
        return T.Compose([
            self.processor,
            MediumNpyTransforms(self.train, self.img_size)
        ])


class MediumNpyTransforms(AugmentationBase):

    # ImageNet
    MEANS = [0.485, 0.456, 0.406]
    STDS  = [0.229, 0.224, 0.225]

    def __init__(self, train, img_size):
        self.img_size = img_size
        super().__init__(train)

    def build_transforms(self):
        return T.Compose([
            T.ToPILImage(),
            T.RandomRotation(degrees=180),
            T.RandomResizedCrop(self.img_size, scale=(0.8, 1), ratio=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(self.MEANS, self.STDS),
            Cutout(self.img_size, length=self.img_size // 4)
        ])


class HeavyNpyTransforms(AugmentationBase):

    # ImageNet
    MEANS = [0.485, 0.456, 0.406]
    STDS  = [0.229, 0.224, 0.225]

    def __init__(self, train, img_size):
        self.img_size = img_size
        super().__init__(train)

    def build_transforms(self):
        return T.Compose([
            T.ToPILImage(),
            T.RandomAffine(
                degrees=180,
                translate=(0, 0.05),
                shear=(-0.05, 0.05)
            ),
            T.RandomResizedCrop(self.img_size, scale=(0.8, 1), ratio=(0.9, 1.1)),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2),
            T.ToTensor(),
            T.Normalize(self.MEANS, self.STDS),
            T.RandomErasing(
                p=0.8,
                scale=(0.05, 0.15),
                ratio=(0.4, 2.5)
            )
        ])


class Cutout:
    """
    Randomly mask out one or more patches from an image.
    https://github.com/uoguelph-mlrg/Cutout

    Parameters
    ----------
    img_size : int
        The size of the images to expect.
    n_holes : int
        Number of patches to cut out of each image.
    length : int
        The length (in pixels) of each square patch.
    """
    def __init__(self, img_size, n_holes=1, length=8):
        self.img_size = img_size
        self.n_holes = n_holes
        self.length = length

    def random_int(self, n):
        return torch.randint(n, (1,))[0].item()

    def random_mask(self):
        h = self.img_size
        w = self.img_size

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = self.random_int(h)
            x = self.random_int(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        return mask

    def __call__(self, img):
        """
        Paramters
        ---------
        img : `torch.Tensor`
            Tensor image of size (C, H, W).

        Returns
        -------
        `torch.Tensor`
            Image with n_holes of dimension length x length cut out of it.
        """
        mask = self.random_mask()
        mask = mask.expand_as(img)
        img = img * mask
        return img
