from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


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
        return self.transform(Image.open(filename))

    def __getitem__(self, index):
        filename = self.df.iloc[index]['filename']
        x = self.load_img(filename)
        if not self.train:
            return x
        y = self.df.iloc[index]['diagnosis']
        return (x, y)

    def __len__(self):
        return self.df.shape[0]
