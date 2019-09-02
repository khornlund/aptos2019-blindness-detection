import numpy as np
import pandas as pd
from torch.utils.data.sampler import BatchSampler

from aptos.utils import setup_logger


class SamplerFactory:
    """
    Factory class to create a `ClassWeightedBatchSampler`.
    """

    def __init__(self, verbose=0):
        self.logger = setup_logger(self, verbose)

    def get(self, df, candidate_idx, batch_size, n_batches, alpha):
        """
        Parameters
        ----------
        df : `pandas.DataFrame`
            A dataframe containing the contents of `train.csv`. Must have `diagnosis` column.

        candidate_idx : array-like of ints
            List of candidate indices to use. May not contain all the indices of `df` if some have
            been reserved for validation.

        batch_size : int
            The batch size to use.

        n_batches : int
            The number of batches per epoch.

        alpha : numeric in range [0, 1]
            Weighting term used to determine weights of each class in each batch.
            When `alpha` == 0, the batch class distribution will approximate the training population
            class distribution.
            When `alpha` == 1, the batch class distribution will approximate a uniform distribution,
            with equal number of samples from each class.
            See :method:`balance_weights` for implementation details.
        """
        self.logger.info('Creating `ClassWeightedBatchSampler`...')
        try:
            n_classes = pd.unique(df['diagnosis']).shape[0]
        except KeyError as ex:
            msg = f'Caught KeyError reading {df.head()}: {ex}'
            self.logger.critical(msg)
            raise Exception(msg)

        class_idxs = []
        for c in range(n_classes):
            idxs = df.iloc[candidate_idx].loc[df['diagnosis'] == c, :].index.values
            class_idxs.append(idxs)

        n_samples = len(candidate_idx)
        class_sizes = np.asarray([len(idxs) for idxs in class_idxs])
        original_weights = np.asarray([size / n_samples for size in class_sizes])
        uniform_weights = np.repeat(1 / n_classes, n_classes)

        self.logger.info(f'Sample population class examples: {class_sizes}')
        self.logger.info(f'Sample population class distribution: {original_weights}')

        weights = self.balance_weights(uniform_weights, original_weights, alpha)
        class_samples_per_batch = self.batch_statistics(weights, class_sizes, batch_size, n_batches)

        return ClassWeightedBatchSampler(class_samples_per_batch, class_idxs, n_batches)

    def balance_weights(self, weight_a, weight_b, alpha):
        assert alpha >= 0 and alpha <= 1, f'invalid alpha {alpha}, must be 0 <= alpha <= 1'
        beta = 1 - alpha
        weights = (alpha * weight_a) + (beta * weight_b)
        self.logger.info(f'Target batch class distribution {weights} using alpha={alpha}')
        return weights

    def batch_statistics(self, weights, class_sizes, batch_size, n_batches):
        """
        Calculates the number of samples of each class to include in each batch, and the number
        of batches required to use all the data in an epoch.
        """
        class_samples_per_batch = np.round((weights * batch_size)).astype(int)

        # cleanup rounding edge-cases
        remainder = batch_size - class_samples_per_batch.sum()
        largest_class = np.argmax(class_samples_per_batch)
        class_samples_per_batch[largest_class] += remainder

        assert class_samples_per_batch.sum() == batch_size

        proportions_of_class_per_batch = class_samples_per_batch / batch_size
        self.logger.info(f'Rounded batch class distribution {proportions_of_class_per_batch}')

        proportions_of_samples_per_batch = class_samples_per_batch / class_sizes

        self.logger.info(f'Expecting {class_samples_per_batch} samples of each class per batch, '
                         f'over {n_batches} batches of size {batch_size}')

        oversample_rates = proportions_of_samples_per_batch * n_batches
        self.logger.info(f'Sampling rates: {oversample_rates}')

        return class_samples_per_batch


class ClassWeightedBatchSampler(BatchSampler):
    """
    Ensures each batch contains a given class distribution.

    The lists of indices for each class are shuffled at the start of each call to `__iter__`.

    Parameters
    ----------
    class_samples_per_batch : `numpy.array(int)`
        The number of samples of each class to include in each batch.

    class_idxs : 2D list of ints
        The indices that correspond to samples of each class.

    n_batches : int
        The number of batches to yield.
    """

    def __init__(self, class_samples_per_batch, class_idxs, n_batches):
        self.class_samples_per_batch = class_samples_per_batch
        self.class_idxs = [CircularList(idx) for idx in class_idxs]
        self.n_batches = n_batches

        self.n_classes = len(self.class_samples_per_batch)
        self.batch_size = self.class_samples_per_batch.sum()

        assert len(self.class_samples_per_batch) == len(self.class_idxs)
        assert isinstance(self.n_batches, int)

    def _get_batch(self, start_idxs):
        selected = []
        for c, size in enumerate(self.class_samples_per_batch):
            selected.extend(self.class_idxs[c][start_idxs[c]:start_idxs[c] + size])
        np.random.shuffle(selected)
        return selected

    def __iter__(self):
        [cidx.shuffle() for cidx in self.class_idxs]
        start_idxs = np.zeros(self.n_classes, dtype=int)
        for bidx in range(self.n_batches):
            yield self._get_batch(start_idxs)
            start_idxs += self.class_samples_per_batch

    def __len__(self):
        return self.n_batches


class CircularList:
    """
    Applies modulo function to indexing.
    """
    def __init__(self, items):
        self._items = items
        self._mod = len(self._items)
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self._items)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(key.start, key.stop)]
        return self._items[key % self._mod]
