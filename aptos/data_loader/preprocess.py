import cv2
import numpy as np
import torchvision.transforms as T

from aptos.utils import setup_logger


class ImgProcessor:

    def __init__(self, data_dir, crop_tol=12, radius_size=300, verbose=0):
        self.data_dir = data_dir
        self.logger = setup_logger(self, verbose)
        self.crop_tol = crop_tol
        self.radius_size = radius_size
        self.sequential = T.Compose([
            self.read_png,
            self.crop_square,
            self.scale_radius,
            self.weighted_blur,
            self.crop_circle,
        ])

    def process(self, filename):
        return self.sequential(filename)

    def read_png(self, filename):
        return cv2.imread(filename)[:, :, ::-1]  # bgr => rgb

    def crop_square(self, img):
        gb = cv2.GaussianBlur(img, (7, 7), 0)
        mask = (gb > self.crop_tol).any(2)
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        return img[y0:y1, x0:x1]

    def scale_radius(self, img):
        x = img[img.shape[0] // 2, :, :].sum(1)
        r = (x > x.mean() / 10).sum() / 2
        s = self.radius_size / r
        return cv2.resize(img, None, fx=s, fy=s)

    def weighted_blur(self, img):
        gb = cv2.GaussianBlur(img, (0, 0), self.radius_size // 30)
        return cv2.addWeighted(img, 4, gb, -4, 128)

    def crop_circle(self, img):
        b = np.zeros(img.shape, dtype=np.uint8)
        cv2.circle(
            b,
            (img.shape[1] // 2, img.shape[0] // 2),
            int(self.radius_size * 0.92),
            (255, 255, 255),
            thickness=-1)
        return img * b + 128 * (1 - b)
