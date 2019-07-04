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

    MEANS = [81.68259022825875 / 255, 87.81281901701134 / 255, 92.54638140735202 / 255]
    STDS  = [51.061115180201085 / 255, 54.74800421817656 / 255, 55.74811655328235 / 255]

    def __init__(self, train, img_size):
        self.img_size = img_size
        super().__init__(train)

    def build_transforms(self):
        return T.Compose([
            T.ToPILImage(),
            T.RandomRotation(degrees=180),
            T.RandomResizedCrop(self.img_size, scale=(0.8, 1), ratio=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(self.MEANS, self.STDS)
        ])
