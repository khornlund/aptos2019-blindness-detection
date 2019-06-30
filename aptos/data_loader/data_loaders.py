from aptos.base import BaseDataLoader

from .datasets import PngDataset
from .augmentation import MediumTransforms


class PngDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers,
                 verbose=0, labels=None, train=True):
        self.data_dir = data_dir
        transform = MediumTransforms(train)
        self.dataset = PngDataset(self.data_dir, transform, train)
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers, verbose=verbose)
