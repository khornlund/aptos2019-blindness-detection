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


class MediumPngTransforms(AugmentationBase):

    MEANS = [0.18345418820778528, 0.2639983979364236, 0.2345641329884529]
    STDS  = [0.11223869025707245, 0.15332063722113767, 0.11037248869736989]

    def __init__(self, train, img_size):
        self.img_size = img_size
        super().__init__(train)

    def build_transforms(self):
        return T.Compose([
            T.RandomAffine(degrees=45, translate=(0.05, 0)),
            T.RandomResizedCrop(self.img_size, scale=(0.8, 1), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(self.MEANS, self.STDS)
        ])


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

    MEANS = [117.49076830707003 / 255, 62.91094476592893 / 255, 20.427623897278952 / 255]
    STDS  = [56.52915067774219 / 255, 31.734976154784277 / 255, 14.035278108570138 / 255]

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
    n_masks : int
        Pre-build N masks to randomly choose from (so they don't have to be generated at runtime).
    """
    def __init__(self, img_size, n_holes=1, length=8, n_masks=int(5e4)):
        self.img_size = img_size
        self.n_holes = n_holes
        self.length = length
        self.n_masks = n_masks

        self.masks = self.get_masks(self.n_masks)

    def get_masks(self, n):
        masks = torch.zeros((n, self.img_size, self.img_size))
        for i in range(n):
            masks[i, :, :] = self.random_mask()
        return masks

    def random_mask(self):
        h = self.img_size
        w = self.img_size

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

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
        mask_idx = np.random.choice(self.masks.shape[0])
        mask = self.masks[mask_idx]
        mask = mask.expand_as(img)
        img = img * mask

        return img
