from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


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
        try:
            return self.transform(str(filename))  # let transforms do loading
        except:
            return torch.zeros((3, 256, 256))

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


class MixupNpyDataset(Dataset):

    N_CLASSES = 5
    train_csv = 'train.csv'

    def __init__(self, data_dir, pre_tsfm, post_tsfm):
        self.pre_tsfm  = pre_tsfm
        self.post_tsfm = post_tsfm

        self.images_dir = Path(data_dir) / 'train_images'
        self.labels_filename = Path(data_dir) / self.train_csv

        self._df = pd.read_csv(self.labels_filename)
        self._df['filename'] = self._df['id_code'].apply(lambda x: self.images_dir / f'{x}.npy')

        self.class_idxs = [
            self.df.loc[self.df['diagnosis'] == c, :].index.values for c in range(self.N_CLASSES)
        ]

    @property
    def df(self):
        return self._df

    def random_float(self):
        return torch.rand((1,))[0].item()

    def random_choice(self, items):
        return items[torch.randint(len(items), (1,))[0].item()]

    def random_neighbour_class(self, c):
        if c == 0:
            return 1
        if c == 4:
            return 3
        return c - 1 if self.random_float() > 0.5 else c + 1

    def load_img(self, filename):
        return self.pre_tsfm(np.load(filename))

    def mixup(self, X1, X2, y1, y2):
        alpha = self.random_float()
        beta = 1 - alpha
        X = Image.blend(X1, X2, alpha)
        y = (alpha * y1) + (beta * y2)
        return X, y

    def __getitem__(self, idx1):
        y1 = self.df.iloc[idx1]['diagnosis']
        y2 = self.random_neighbour_class(y1)

        idx2 = self.random_choice(self.class_idxs[y2])
        assert y2 == self.df.iloc[idx2]['diagnosis']

        f1 = self.df.iloc[idx1]['filename']
        f2 = self.df.iloc[idx2]['filename']

        X1 = self.load_img(f1)
        X2 = self.load_img(f2)

        X, y = self.mixup(X1, X2, y1, y2)
        return (self.post_tsfm(X), y)

    def __len__(self):
        return self.df.shape[0]
