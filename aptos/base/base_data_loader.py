import math

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler

from aptos.utils import setup_logger
from aptos.data_loader import SamplerFactory


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, validation_split, num_workers, weighted=None,
                 collate_fn=default_collate, verbose=0):
        self.logger = setup_logger(self, verbose=verbose)

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)
        self.logger.info(f'Loaded {len(dataset)} images')

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        # get validation indices/sampler
        valid_sampler, valid_idx = self._validation_split(self.n_samples, split)
        self.n_samples -= len(valid_idx)

        # balance classes
        weights = self._weight_classes(self.dataset.df)['weight'].values
        for idx in valid_idx:
            weights[idx] = 0  # set the weight to zero to select validation indices
        train_sampler = WeightedRandomSampler(weights, self.n_samples)

        return train_sampler, valid_sampler

    def _validation_split(self, n_samples, split):
        idx_full = np.arange(self.n_samples)
        self.len_valid = int(self.n_samples * self.validation_split)
        valid_idx = np.random.choice(idx_full, size=self.len_valid, replace=False)
        valid_sampler = SubsetRandomSampler(valid_idx)
        return valid_sampler, valid_idx

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
