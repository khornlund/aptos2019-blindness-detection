import math

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

from aptos.utils import setup_logger


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers,
                 collate_fn=default_collate, verbose=0):
        self.logger = setup_logger(self, verbose=verbose)
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)
        self.logger.info(f'Loaded {len(dataset)} images')

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        # get validation indices and create random subset sampler
        self.len_valid = int(self.n_samples * self.validation_split)
        valid_idx = np.random.choice(idx_full, size=self.len_valid, replace=False)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.n_samples -= self.len_valid

        # balance classes
        weights = self._weight_classes(self.dataset.df)['weight'].values
        for idx in valid_idx:
            weights[idx] = 0  # set the weight to zero to select validation indices
        train_sampler = WeightedRandomSampler(weights, self.n_samples)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        return train_sampler, valid_sampler

    def _weight_classes(self, df):
        n = df.shape[0]
        self.logger.info(f'Adjusting weights for {n} samples')
        grp_df = df.groupby('diagnosis').count()
        grp_df['weight'] = grp_df['id_code'].apply(lambda x: math.sqrt(n / x))
        grp_df['projected'] = grp_df['id_code'] * grp_df['weight']
        self.logger.info(f'Weight adjusted: {grp_df}')
        weight_df = df.join(grp_df[['weight']], on='diagnosis')
        return weight_df

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
