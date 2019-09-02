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

    def __init__(self, data_dir, train_tsfm, test_tsfm, train=True):
        self.train = train
        self.train_tsfm = train_tsfm
        self.test_tsfm = test_tsfm

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
        if self.train:
            return self.train_tsfm(np.load(filename))
        return self.test_tsfm(np.load(filename))

    def __getitem__(self, index):
        filename = self.df.iloc[index]['filename']
        x = self.load_img(filename)
        y = self.df.iloc[index]['diagnosis']
        return (x, y)

    def __len__(self):
        return self.df.shape[0]
