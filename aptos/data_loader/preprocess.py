import cv2
import numpy as np
import torchvision.transforms as T

from aptos.utils import setup_logger


class ImgProcessor:
    """
    This class is responsible for preprocessing the images, eg. crop, sharpen, resize, normalise.
    """

    def __init__(self, crop_tol=12, img_width=512, verbose=0):
        self.logger = setup_logger(self, verbose)
        self.crop_tol = crop_tol
        self.img_width = img_width
        self.sequential = T.Compose([
            self.read_png,
            self.crop_box,
            self.resize,
            self.crop_circle
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
        new_height = min(self.img_width, int(H * scale_percent))
        dim = (self.img_width, new_height)
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    def crop_circle(self, img):
        """
        Apply a circular crop to remove edge effects.
        """
        H, W, C = img.shape
        circle_img = np.zeros((H, W), dtype=np.uint8)
        x, y = W // 2, H // 2
        r = int(W * 0.95 / 2)  # cut a small amount off
        cv2.circle(circle_img, (x, y), r, 1, thickness=-1)
        return cv2.bitwise_and(img, img, mask=circle_img)

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

    # def pad_square(self, img):
    #     """
    #     Pad the top/bottom of the image with zeros to make it square.
    #     """
    #     try:
    #         H, W, C = img.shape
    #         assert H <= W
    #         if H == W:
    #             return img
    #         pad_amount = W - H
    #         top_pad = pad_amount // 2
    #         btm_pad = pad_amount - top_pad
    #         return cv2.copyMakeBorder(img, top_pad, btm_pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    #     except Exception as ex:
    #         msg = f'Caught exception: {ex}'
    #         print(msg)
    #         raise Exception(msg)