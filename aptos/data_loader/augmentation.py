import torchvision.transforms as T


class AugmentationBase:

    MEANS = [0.18345418820778528, 0.2639983979364236, 0.2345641329884529]
    STDS  = [0.11223869025707245, 0.15332063722113767, 0.11037248869736989]

    def __init__(self, train):
        self.train = train
        self.transform = self.build_transforms()

    def build_transforms(self, images):
        raise NotImplementedError('Not implemented!')

    def __call__(self, images):
        return self.transform(images)


class MediumTransforms(AugmentationBase):

    def __init__(self, train):
        super().__init__(train)

    def build_transforms(self):
        if not self.train:
            return T.Compose([
                T.ToTensor(),
                T.Normalize(self.MEANS, self.STDS)
            ])

        return T.Compose([
            T.RandomAffine(degrees=45, translate=(0.05, 0)),
            T.RandomResizedCrop(512, scale=(0.8, 1), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(self.MEANS, self.STDS)
        ])
