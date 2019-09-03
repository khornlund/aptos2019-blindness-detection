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

    # # ImageNet
    # MEANS = [0.485, 0.456, 0.406]
    # STDS  = [0.229, 0.224, 0.225]

    MEANS = [0.5, 0.5, 0.5]
    STDS  = [0.075, 0.075, 0.075]

    def __init__(self, train, img_size):
        self.img_size = img_size
        super().__init__(train)

    def build_transforms(self):
        return T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomResizedCrop(self.img_size, scale=(0.8, 1)),
            T.ToTensor(),
            T.Normalize(self.MEANS, self.STDS),
        ])


class HeavyNpyTransforms(AugmentationBase):

    # # ImageNet
    # MEANS = [0.485, 0.456, 0.406]
    # STDS  = [0.229, 0.224, 0.225]

    MEANS = [0.5, 0.5, 0.5]
    STDS  = [0.075, 0.075, 0.075]

    def __init__(self, train, img_size):
        self.img_size = img_size
        super().__init__(train)

    def build_transforms(self):
        return T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(
                degrees=180,
                translate=(0.07, 0.0),
                shear=(0.05),
                fillcolor=(128, 128, 128)
            ),
            T.RandomResizedCrop(self.img_size, scale=(0.8, 1)),
            T.ToTensor(),
            T.Normalize(self.MEANS, self.STDS),
            T.RandomErasing(
                p=0.8,
                scale=(0.05, 0.15),
                ratio=(0.4, 2.5)
            )
        ])
