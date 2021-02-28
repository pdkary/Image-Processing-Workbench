import numpy as np
from src.image_utils import ImageUtils
from src.fourier import FourierTransform
from src.mask_filter import MaskFilter
from src.kernels import *
import cv2


class Convolution:
    @staticmethod
    def convolve(f_img, g_img, filename=None):
        new_img = np.ndarray(f_img.shape)
        if len(f_img.shape) == 3:
            new_img = Convolution.convolve_color(f_img, g_img, filename)
        if len(f_img.shape) == 2:
            new_img = Convolution.convolve_flat(f_img, g_img, filename)

        if filename is not None:
            cv2.imwrite("images/convolution/" + filename + ".jpg", new_img)
        return new_img

    @staticmethod
    def convolve_color(f_img, g_img, filename=None):
        new_img = np.ndarray(shape=f_img.shape, dtype=f_img.dtype)
        if len(g_img.shape) == 3:
            kR, kG, kB = g_img[:, :, 0], g_img[:, :, 1], g_img[:, :, 2]
        else:
            kR, kG, kB = g_img, g_img, g_img
        new_img[:, :, 0] = Convolution.convolve_flat(f_img[:, :, 0], kR)
        new_img[:, :, 1] = Convolution.convolve_flat(f_img[:, :, 1], kG)
        new_img[:, :, 2] = Convolution.convolve_flat(f_img[:, :, 2], kB)
        return new_img

    @staticmethod
    def convolve_flat(f_img, g_img, filename=None):
        g_img = np.fliplr(g_img)
        g_img = np.flipud(g_img)
        dx = f_img.shape[0] - g_img.shape[0]
        dy = f_img.shape[1] - g_img.shape[1]
        bx, by = dx // 2, dy // 2
        g_img = ImageUtils.pad_to_size(g_img, f_img.shape)

        ft_f = FourierTransform.transform(f_img)
        ft_g = FourierTransform.transform(g_img)
        ft_fg = ft_f * ft_g
        conv_fg = FourierTransform.inverse_transform(ft_fg)
        if filename is not None:
            cv2.imwrite(
                "images/convolution/" + filename + ".jpg",
                FourierTransform.get_viewable_fourier(ft_fg),
            )
        return conv_fg
    