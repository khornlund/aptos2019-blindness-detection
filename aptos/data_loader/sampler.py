import math

import numpy as np
import pandas as pd
from torch.utils.data.sampler import BatchSampler


class SamplerFactory:

    @classmethod
    def get(cls, df, candidate_idx, batch_size, alpha):
        assert alpha >= 0 and alpha <= 1, f'invalid alpha {alpha}, must be 0 <= alpha <= 1'

        n_classes = pd.unique(df['diagnosis']).shape[0]

        class_idxs = []
        for c in range(n_classes):
            idxs = df.iloc[candidate_idx].loc[df['diagnosis'] == c, :].index.values
            class_idxs.append(idxs)

        n_samples = len(candidate_idx)
        class_sizes = np.asarray([len(idxs) for idxs in class_idxs])
        uniform_weights = np.repeat(1 / n_classes, n_classes)
        original_weights = np.asarray([size / n_samples for size in class_sizes])

        weights = cls.balance_weights(original_weights, uniform_weights, alpha)
        n_batches = cls.n_batches(weights, class_sizes, batch_size)

        return ClassWeightedBatchSampler(weights, class_idxs, batch_size, n_batches)

    @classmethod
    def balance_weights(cls, weight_a, weight_b, alpha):
        beta = 1 - alpha
        return (alpha * weight_a) + (beta * weight_b)

    @classmethod
    def n_batches(cls, weights, class_sizes, batch_size):
        proportions_per_batch = (weights * batch_size) / class_sizes
        n_batches = math.ceil(1 / min(proportions_per_batch))
        return n_batches


class ClassWeightedBatchSampler(BatchSampler):
    """
    Ensures each batch contains a given class distribution.

    Parameters
    ----------
    class_batch_sizes : array-like of int
        The number of indices to select of each class.

    class_idxs : 2D list of ints
        The indices that correspond to samples of each class.

    Example
    -------
    .. code::

        class_batch_sizes = [2, 3, 4]
        class_idxs = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14]
        ]

        sampler = ClassWeightedSampler(class_batch_sizes, class_idxs)
        idxs = list([idx for idx in sampler])

        # idxs will contain 2 of [0, 1, 2, 3, 4],
        #                   3 of [5, 6, 7, 8, 9], and
        #                   4 of [10, 11, 12, 13, 14]
        # in a random order.
    """

    def __init__(self, class_weights, class_idxs, batch_size, n_batches):
        self.class_weights = class_weights
        self.class_idxs = [CircularList(idx) for idx in class_idxs]
        self.batch_size = batch_size
        self.n_batches = n_batches

        self.n_classes = len(self.class_weights)
        self.class_sizes = np.asarray([batch_size * w for w in self.class_weights], dtype=int)

        # handle rounding edge cases
        remainder = self.batch_size - self.class_sizes.sum()
        self.class_sizes[0] += remainder

        assert isinstance(self.batch_size, int)
        assert isinstance(self.n_batches, int)

    def _get_batch(self, start_idxs):
        selected = []
        for c, size in enumerate(self.class_sizes):
            selected.extend(self.class_idxs[c][start_idxs[c]:start_idxs[c] + size])
        np.random.shuffle(selected)
        return selected

    def __iter__(self):
        [cidx.shuffle() for cidx in self.class_idxs]
        start_idxs = np.zeros(self.n_classes, dtype=int)
        for bidx in range(self.n_batches):
            yield self._get_batch(start_idxs)
            start_idxs += self.class_sizes

    def __len__(self):
        return self.n_batches


class CircularList:
    """
    Applies modulo function to indexing.
    """
    def __init__(self, items):
        self._items = items
        self._mod = len(self._items)

    def shuffle(self):
        np.random.shuffle(self._items)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(key.start, key.stop)]
        return self._items[key % self._mod]
