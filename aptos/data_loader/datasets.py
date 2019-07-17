from pathlib import Path

from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class PngDataset(Dataset):

    train_csv = 'train.csv'
    test_csv  = 'test.csv'

    def __init__(self, data_dir, transform, train=True):
        self.train = train
        self.transform = transform

        if self.train:
            self.images_dir = Path(data_dir) / 'train_images'
            self.labels_filename = Path(data_dir) / self.train_csv
        else:
            self.images_dir = Path(data_dir) / 'test_images'
            self.labels_filename = Path(data_dir) / self.test_csv

        self._df = pd.read_csv(self.labels_filename)
        self._df['filename'] = self._df['id_code'].apply(lambda x: self.images_dir / f'{x}.png')

    @property
    def df(self):
        return self._df

    def load_img(self, filename):
        return self.transform(str(filename))  # let transforms do loading

    def __getitem__(self, index):
        filename = self.df.iloc[index]['filename']
        x = self.load_img(filename)
        if not self.train:
            return x
        y = self.df.iloc[index]['diagnosis']
        return (x, y)

    def __len__(self):
        return self.df.shape[0]


class NpyDataset(Dataset):

    train_csv = 'train.csv'
    test_csv  = 'test.csv'

    def __init__(self, data_dir, transform, train=True):
        self.train = train
        self.transform = transform

        if self.train:
            self.images_dir = Path(data_dir) / 'train_images'
            self.labels_filename = Path(data_dir) / self.train_csv
        else:
            self.images_dir = Path(data_dir) / 'test_images'
            self.labels_filename = Path(data_dir) / self.test_csv

        self._df = pd.read_csv(self.labels_filename)
        self._df['filename'] = self._df['id_code'].apply(lambda x: self.images_dir / f'{x}.npy')

    @property
    def df(self):
        return self._df

    def load_img(self, filename):
        return self.transform(np.load(filename))

    def __getitem__(self, index):
        filename = self.df.iloc[index]['filename']
        x = self.load_img(filename)
        if not self.train:
            return x
        y = self.df.iloc[index]['diagnosis']
        return (x, y)

    def __len__(self):
        return self.df.shape[0]


class NpyDiagnosisDataset(Dataset):

    train_csv = 'train.csv'
    test_csv  = 'test.csv'

    def __init__(self, diagnosis, data_dir, transform):
        self.diagnosis = diagnosis
        self.transform = transform

        self.images_dir = Path(data_dir) / 'train_images'
        self.labels_filename = Path(data_dir) / self.train_csv

        self._df = pd.read_csv(self.labels_filename)
        self._df = self._df.loc[self._df['diagnosis'] == diagnosis, :]
        self._df['filename'] = self._df['id_code'].apply(lambda x: self.images_dir / f'{x}.npy')

    @property
    def df(self):
        return self._df

    def load_img(self, filename):
        return self.transform(np.load(filename))

    def __getitem__(self, index):
        filename = self.df.iloc[index]['filename']
        x = self.load_img(filename)
        y = self.df.iloc[index]['diagnosis']
        return (x, y)

    def __len__(self):
        return self._df.shape[0]


class NpyPairedDataset(Dataset):
    """
    Returns pairs of samples from two datasets.
    """
    def __init__(self, diagnosis_a, diagnosis_b, data_dir, transform):
        dsa = NpyDiagnosisDataset(diagnosis_a, data_dir, transform)
        dsb = NpyDiagnosisDataset(diagnosis_b, data_dir, transform)

        # save s.t. a > b
        if len(dsa) > len(dsb):
            self.dsa = dsa
            self.dsb = dsb
        else:
            self.dsb = dsa
            self.dsa = dsb

        self.idxsa = np.arange(0, len(self.dsa))
        self.idxsb = np.arange(0, len(self.dsb))

    def __getitem__(self, idxb):
        """
        Returns
        -------
        tuple(Tensor_A, Tensor_B, bool)
            Tensors A and B contain images. The bool is true if the diagnosis of A is more
            severe than the diagnosis of B.
        """
        idxa = np.random.choice(self.idxsa)
        data_a, target_a = self.dsa[idxa]
        data_b, target_b = self.dsb[idxb]
        if np.random.random() > 0.5:
            return (data_a, data_b, (data_a.diagnosis > data_b.diagnosis))
        else:
            return (data_b, data_a, (data_b.diagnosis > data_a.diagnosis))

    def __len__(self):
        return len(self.dsb)
