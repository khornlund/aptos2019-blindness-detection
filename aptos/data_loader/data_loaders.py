import copy

import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler, SequentialSampler

from aptos.utils import setup_logger

from .datasets import PngDataset, NpyDataset, MixupNpyDataset
from .augmentation import (InplacePngTransforms, MediumNpyTransforms, HeavyNpyTransforms,
                           MixupNpyTransforms)
from .sampler import SamplerFactory


class DataLoaderBase(DataLoader):

    def __init__(self, dataset, batch_size, epoch_size, validation_split, num_workers,
                 train=True, alpha=None, verbose=0):
        self.verbose = verbose
        self.logger = setup_logger(self, self.verbose)
        self.ids = dataset.df['id_code'].values

        self.sampler, self.valid_sampler = self._setup_samplers(
            dataset,
            batch_size,
            epoch_size,
            validation_split,
            alpha)

        init_kwargs = {
            'dataset': dataset,
            'num_workers': num_workers
        }
        super().__init__(batch_sampler=self.sampler, **init_kwargs)

    def _setup_samplers(self, dataset, batch_size, epoch_size, validation_split, alpha):
        # get sampler & indices to use for validation
        valid_sampler, valid_idx = self._setup_validation(dataset, batch_size, validation_split)

        # get sampler & indices to use for training/testing
        train_sampler, n_samples = self._setup_train(
            dataset, batch_size, epoch_size, alpha, valid_idx)
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

    def _setup_train(self, dataset, batch_size, epoch_size, alpha, exclude_idx):
        all_idx = np.arange(len(dataset))
        train_idx = [i for i in all_idx if i not in exclude_idx]

        if alpha is None:
            self.logger.info('No sample weighting selected.')
            subset = Subset(dataset, train_idx)
            sampler = BatchSampler(SequentialSampler(subset), batch_size, False)
            return sampler, len(train_idx)

        factory = SamplerFactory(self.verbose)
        sampler = factory.get(dataset.df, train_idx, batch_size, epoch_size, alpha)
        return sampler, len(sampler) * batch_size

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            dataset = copy.deepcopy(self.dataset)
            dataset.train = False
            init_kwargs = {
                'dataset': dataset,
                'num_workers': self.num_workers
            }
            return DataLoader(batch_sampler=self.valid_sampler, **init_kwargs)


class PngDataLoader(DataLoaderBase):

    def __init__(self, data_dir, batch_size, validation_split, num_workers, img_size,
                 train=True, alpha=None, verbose=0):
        transform = InplacePngTransforms(train, img_size)
        dataset = PngDataset(data_dir, transform, train=train)

        super().__init__(dataset, batch_size, None, validation_split, num_workers,
                         train=train, alpha=alpha, verbose=verbose)


class NpyDataLoader(DataLoaderBase):

    def __init__(self, data_dir, batch_size, epoch_size, validation_split, num_workers, img_size,
                 train=True, alpha=None, verbose=0):
        train_tsfm = HeavyNpyTransforms(True, img_size)
        test_tsfm  = MediumNpyTransforms(False, img_size)
        dataset = NpyDataset(data_dir, train_tsfm, test_tsfm, train=train)

        super().__init__(dataset, batch_size, epoch_size, validation_split, num_workers,
                         train=train, alpha=alpha, verbose=verbose)


class MixupDataLoader(DataLoader):

    def __init__(self, data_dir, batch_size, epoch_size, validation_split, num_workers, img_size,
                 mixup=0.4, alpha=None, verbose=0):
        self.logger = setup_logger(self, verbose)

        if validation_split > 0:
            valid_tsfm = MediumNpyTransforms(train=False, img_size=img_size)
            dataset_valid = NpyDataset(data_dir, valid_tsfm, None, train=True)

            all_idx = np.arange(len(dataset_valid))
            len_valid = int(len(all_idx) * validation_split)
            valid_idx = np.random.choice(all_idx, size=len_valid, replace=False)
            valid_sampler = BatchSampler(SubsetRandomSampler(valid_idx), batch_size, False)
            self.logger.info(f'Selected {len(valid_idx)}/{len(all_idx)} indices for validation')
            valid_targets = dataset_valid.df.iloc[valid_idx].groupby('diagnosis').count()
            self.logger.info(f'Validation class distribution: {valid_targets}')

            self._loader_valid = DataLoader(
                dataset_valid,
                batch_sampler=valid_sampler,
                num_workers=num_workers
            )
        else:
            valid_idx = []

        train_tsfm = MixupNpyTransforms(train=True, img_size=img_size)
        dataset_train = MixupNpyDataset(data_dir, train_tsfm, alpha=mixup, train=True)
        all_idx = np.arange(len(dataset_train))
        train_idx = [i for i in all_idx if i not in valid_idx]

        if alpha is None:
            self.logger.info('No sample weighting selected.')
            subset = Subset(dataset_train, train_idx)
            sampler = BatchSampler(SequentialSampler(subset), batch_size, False)
            return sampler, len(train_idx)

        factory = SamplerFactory(verbose)
        sampler = factory.get(dataset_train.df, train_idx, batch_size, epoch_size, alpha)
        self.n_samples = len(sampler) * batch_size

        super().__init__(dataset_train, batch_sampler=sampler, num_workers=num_workers)

    def split_validation(self):
        if self._loader_valid is None:
            return None
        else:
            return self._loader_valid
