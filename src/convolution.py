import numpy as np
from src.image_utils import ImageUtils
from src.fourier import FourierTransform
from src.mask_filter import MaskFilter
from src.kernels import *
import cv2


class Convolution:
    @staticmethod
    def convolve(f_img, g_img, convolve_g=True, filename=None, inverse=False):
        new_img = np.ndarray(f_img.shape)
        if len(f_img.shape) == 3:
            new_img = Convolution.convolve_color(f_img, g_img, convolve_g, filename,inverse)
        if len(f_img.shape) == 2:
            new_img = Convolution.convolve_flat(f_img, g_img, convolve_g, filename,inverse)

        if filename is not None:
            cv2.imwrite("images/convolution/" + filename + ".jpg", new_img)
        return new_img

    @staticmethod
    def convolve_color(f_img, g_img,convolve_g=True, filename=None,inverse=False):
        new_img = np.ndarray(shape=f_img.shape, dtype=f_img.dtype)
        fR = filename+"_R" if filename is not None else None 
        fG = filename+"_G" if filename is not None else None 
        fB = filename+"_B" if filename is not None else None 
        if len(g_img.shape) == 3:
            kR, kG, kB = g_img[:, :, 0], g_img[:, :, 1], g_img[:, :, 2]
        else:
            kR, kG, kB = g_img, g_img, g_img
        if inverse:
            new_img[:, :, 0] = Convolution.convolve_flat(f_img[:, :, 0], kR,convolve_g=convolve_g,filename=fR,inverse=True)
            new_img[:, :, 1] = Convolution.convolve_flat(f_img[:, :, 1], kG,convolve_g=convolve_g,filename=fG,inverse=True)
            new_img[:, :, 2] = Convolution.convolve_flat(f_img[:, :, 2], kB,convolve_g=convolve_g,filename=fB,inverse=True)
        else:
            new_img[:, :, 0] = Convolution.convolve_flat(f_img[:, :, 0], kR,convolve_g=convolve_g,filename=fR,inverse=False)
            new_img[:, :, 1] = Convolution.convolve_flat(f_img[:, :, 1], kG,convolve_g=convolve_g,filename=fG,inverse=False)
            new_img[:, :, 2] = Convolution.convolve_flat(f_img[:, :, 2], kB,convolve_g=convolve_g,filename=fB,inverse=False)
        return new_img

    @staticmethod
    def convolve_flat(f_img, g_img, convolve_g=True, filename=None,inverse=False):
        g_img = np.fliplr(g_img)
        g_img = np.flipud(g_img)
        g_img = ImageUtils.pad_to_size(g_img, f_img.shape)

        ft_f = FourierTransform.transform(f_img)
        if convolve_g:
            ft_g = FourierTransform.transform(g_img)
        else:
            ft_g = g_img
        if inverse:
            ft_fg = ft_f / ft_g
        else:
            ft_fg = ft_f * ft_g
        if convolve_g:
            ft_fg = FourierTransform.reshape_fourier(ft_fg)
        conv_fg = FourierTransform.inverse_transform(ft_fg)
        
        if filename is not None:
            cv2.imwrite(
                "images/convolution/" + filename + ".jpg",
                FourierTransform.get_viewable_fourier(ft_fg),
            )
        return conv_fg
    