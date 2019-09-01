import cv2
import numpy as np
import torchvision.transforms as T

from aptos.utils import setup_logger


class ImgProcessor:
    """
    This class is responsible for preprocessing the images, eg. crop, sharpen, resize, normalise.
    """

    def __init__(self, crop_tol=12, img_width=300, verbose=0):
        self.logger = setup_logger(self, verbose)
        self.crop_tol = crop_tol
        self.img_width = img_width
        self.sequential = T.Compose([
            self.read_png,
            self.crop_box,
            self.resize,
            # self.pad_square
        ])

    def __call__(self, filename):
        return self.sequential(filename)

    def read_png(self, filename):
        """
        Load the image into a numpy array, and switch the channel order so it's in the format
        expected by matplotlib (rgb).
        """
        return cv2.imread(filename)[:, :, ::-1]  # bgr => rgb

    def crop_box(self, img):
        """
        Apply a bounding box to crop empty space around the image. In order to find the bounding
        box, we blur the image and then apply a threshold. The blurring helps avoid the case where
        an outlier bright pixel causes the bounding box to be larger than it needs to be.
        """
        gb = cv2.GaussianBlur(img, (7, 7), 0)
        mask = (gb > self.crop_tol).any(2)
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        return img[y0:y1, x0:x1]

    def resize(self, img):
        H, W, C = img.shape
        scale_percent = self.img_width / W
        dim = (self.img_width, int(H * scale_percent))
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    def pad_square(self, img):
        """
        Pad the top/bottom of the image with zeros to make it square.
        """
        H, W, C = img.shape
        if H == W:
            return img
        top_pad = (H - self.img_width) // 2
        btm_pad = self.img_width - top_pad
        return np.pad(img, ((0, 0), (top_pad, btm_pad), (0, 0)), mode='constant')

    # -- unused --

    # def scale_radius(self, img):
    #     """
    #     Resize the image so the radius is `self.radius_size` pixels.
    #     """
    #     x = img[img.shape[0] // 2, :, :].sum(1)
    #     r = (x > x.mean() / 10).sum() / 2
    #     s = self.radius_size / r
    #     return cv2.resize(img, None, fx=s, fy=s)

    # def sharpen(self, img):
    #     """
    #     Sharpen the image by subtracting a gaussian blur.
    #     """
    #     ksize = (0, 0)
    #     sigmaX = self.radius_size // 30
    #     alpha = 4
    #     beta = -4
    #     gamma = 255 // 2 + 1
    #     gb = cv2.GaussianBlur(img, ksize, sigmaX)
    #     return cv2.addWeighted(img, alpha, gb, beta, gamma)

    # def crop_circle(self, img):
    #     """
    #     Apply a circular crop to remove edge effects.
    #     """
    #     b = np.zeros(img.shape, dtype=np.uint8)
    #     cv2.circle(
    #         b,
    #         (img.shape[1] // 2, img.shape[0] // 2),
    #         int(self.radius_size * 0.92),
    #         (255, 255, 255),
    #         thickness=-1)
    #     return img * b  # + 128 * (1 - b)
