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

    MEANS = [0.6193246715450339, 0.5676388422333433, 0.5303413730576545]
    STDS  = [0.12337693906775953, 0.09914381078783173, 0.06671092824144163]

    def __init__(self, train, img_size):
        self.img_size = img_size
        super().__init__(train)

    def build_transforms(self):
        return T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomResizedCrop(self.img_size, scale=(0.9, 0.98)),
            T.ToTensor(),
            T.Normalize(self.MEANS, self.STDS),
        ])


class HeavyNpyTransforms(AugmentationBase):

    MEANS = [0.6193246715450339, 0.5676388422333433, 0.5303413730576545]
    STDS  = [0.12337693906775953, 0.09914381078783173, 0.06671092824144163]

    def __init__(self, train, img_size):
        self.img_size = img_size
        super().__init__(train)

    def build_transforms(self):
        return T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            # T.RandomAffine(
            #     degrees=5,
            #     translate=(0.03, 0.0),
            #     # shear=(0.05),
            #     # fillcolor=(128, 128, 128)
            # ),
            T.RandomResizedCrop(self.img_size, scale=(0.8, 0.95)),
            T.ToTensor(),
            T.Normalize(self.MEANS, self.STDS),
            T.RandomErasing(
                p=0.8,
                scale=(0.05, 0.15),
                ratio=(0.4, 2.5)
            )
        ])


class MixupNpyTransforms(AugmentationBase):

    MEANS = [0.6193246715450339, 0.5676388422333433, 0.5303413730576545]
    STDS  = [0.12337693906775953, 0.09914381078783173, 0.06671092824144163]

    def __init__(self, train, img_size):
        self.img_size = img_size
        super().__init__(train)

    def build_transforms(self):
        return T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomResizedCrop(self.img_size, scale=(0.9, 0.98)),
            T.ToTensor(),
            T.Normalize(self.MEANS, self.STDS),
            T.RandomErasing(
                p=0.20,
                scale=(0.03, 0.08),
                ratio=(0.5, 2.0)
            )
        ])
