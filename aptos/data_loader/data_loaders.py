from aptos.base import BaseDataLoader

from .datasets import PngDataset
from .augmentation import MediumTransforms


class PngDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, img_size,
                 verbose=0, train=True):
        self.data_dir = data_dir
        transform = MediumTransforms(train, img_size)
        self.dataset = PngDataset(self.data_dir, transform, train=train)
        self.ids = self.dataset.df['id_code'].values
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers, verbose=verbose)
