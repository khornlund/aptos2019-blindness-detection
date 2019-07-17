import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler, SequentialSampler

from aptos.utils import setup_logger

from .datasets import PngDataset, NpyDataset, NpyPairedDataset
from .augmentation import InplacePngTransforms, MediumNpyTransforms
from .sampler import SamplerFactory


class DataLoaderBase(DataLoader):

    def __init__(self, dataset, batch_size, validation_split, num_workers,
                 train=True, alpha=None, verbose=0):
        self.verbose = verbose
        self.logger = setup_logger(self, self.verbose)
        self.ids = dataset.df['id_code'].values

        self.sampler, self.valid_sampler = self._setup_samplers(
            dataset,
            batch_size,
            validation_split,
            alpha)

        self.init_kwargs = {
            'dataset': dataset,
            'num_workers': num_workers
        }
        super().__init__(batch_sampler=self.sampler, **self.init_kwargs)

    def _setup_samplers(self, dataset, batch_size, validation_split, alpha):
        # get sampler & indices to use for validation
        valid_sampler, valid_idx = self._setup_validation(dataset, batch_size, validation_split)

        # get sampler & indices to use for training/testing
        train_sampler, n_samples = self._setup_train(dataset, batch_size, alpha, valid_idx)
        self.n_samples = n_samples

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
        valid_targets = dataset.df.iloc[valid_idx].groupby('diagnosis').count()
        self.logger.info(f'Validation class distribution: {valid_targets}')
        return valid_sampler, valid_idx

    def _setup_train(self, dataset, batch_size, alpha, exclude_idx):
        all_idx = np.arange(len(dataset))
        train_idx = [i for i in all_idx if i not in exclude_idx]

        if alpha is None:
            self.logger.info('No sample weighting selected.')
            subset = Subset(dataset, train_idx)
            sampler = BatchSampler(SequentialSampler(subset), batch_size, False)
            return sampler, len(train_idx)

        factory = SamplerFactory(self.verbose)
        sampler = factory.get(dataset.df, train_idx, batch_size, alpha)
        return sampler, len(sampler) * batch_size

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(batch_sampler=self.valid_sampler, **self.init_kwargs)


class PngDataLoader(DataLoaderBase):

    def __init__(self, data_dir, batch_size, validation_split, num_workers, img_size,
                 train=True, alpha=None, verbose=0):
        transform = InplacePngTransforms(train, img_size)
        dataset = PngDataset(data_dir, transform, train=train)

        super().__init__(dataset, batch_size, validation_split, num_workers,
                         train=train, alpha=alpha, verbose=verbose)


class NpyDataLoader(DataLoaderBase):

    def __init__(self, data_dir, batch_size, validation_split, num_workers, img_size,
                 train=True, alpha=None, verbose=0):
        transform = MediumNpyTransforms(train, img_size)
        dataset = NpyDataset(data_dir, transform, train=train)

        super().__init__(dataset, batch_size, validation_split, num_workers,
                         train=train, alpha=alpha, verbose=verbose)


class NpyPairedDataLoader(DataLoader):

    def __init__(self, data_dir, batch_size, validation_split, num_workers, img_size,
                 train=True, verbose=0):
        self.logger = setup_logger(self, verbose)
        transform = MediumNpyTransforms(train, img_size)
        dataset = NpyPairedDataset(0, 4, data_dir, transform)

        self.init_kwargs = {
            'dataset': dataset,
            'shuffle': True,
            'batch_size': batch_size,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)
