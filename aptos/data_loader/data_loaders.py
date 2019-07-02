import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler

from aptos.utils import setup_logger

from .datasets import PngDataset
from .augmentation import MediumTransforms
from .sampler import SamplerFactory


class PngDataLoader(DataLoader):

    def __init__(self, data_dir, batch_size, validation_split, num_workers, img_size,
                 train=True, weighted=None, verbose=0):
        self.data_dir = data_dir
        self.logger = setup_logger(self, verbose=verbose)

        transform = MediumTransforms(train, img_size)
        dataset = PngDataset(self.data_dir, transform, train=train)
        self.ids = dataset.df['id_code'].values

        self.sampler, self.valid_sampler = self._setup_samplers(
            dataset,
            batch_size,
            validation_split,
            weighted)

        self.init_kwargs = {
            'dataset': dataset,
            'num_workers': num_workers
        }
        super().__init__(batch_sampler=self.sampler, **self.init_kwargs)

    def _setup_samplers(self, dataset, batch_size, validation_split, weighted):
        # get sampler & indices to use for validation
        valid_sampler, valid_idx = self._setup_validation(dataset, batch_size, validation_split)

        # get sampler & indices to use for training/testing
        train_sampler, train_idx = self._setup_train(dataset, batch_size, weighted, valid_idx)
        self.n_samples = len(train_idx)

        return (train_sampler, valid_sampler)

    def _setup_validation(self, dataset, batch_size, split):
        if split == 0.0:
            self.logger.info('No samples selected for validation.')
            return None, []
        all_idx = np.arange(len(dataset))
        len_valid = int(len(all_idx) * split)
        valid_idx = np.random.choice(all_idx, size=len_valid, replace=False)
        valid_sampler = BatchSampler(SubsetRandomSampler(valid_idx), batch_size, False)
        self.logger.info(f'Selected {len(valid_idx)}/{len(all_idx)} indices for validation')
        return valid_sampler, valid_idx

    def _setup_train(self, dataset, batch_size, weighted, exclude_idx):
        all_idx = np.arange(len(dataset))
        train_idx = [i for i in all_idx if i not in exclude_idx]

        if weighted is None:
            self.logger.info('No sample weighting selected.')
            sampler = BatchSampler(SubsetRandomSampler(train_idx), batch_size, False)
            return sampler, train_idx

        sampler = SamplerFactory.get(dataset.df, train_idx, batch_size, weighted)
        return sampler, train_idx

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(batch_sampler=self.valid_sampler, **self.init_kwargs)
