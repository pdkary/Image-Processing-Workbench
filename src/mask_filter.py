import numpy as np
from math import e, pi
import cv2
from src.fourier import FourierTransform


class MaskFilter:
    def d_to_center(N, M, x, y):
        return np.sqrt((x - N / 2) ** 2 + (y - M / 2) ** 2)

    def apply_mask(img, mask, filename=None):
        if len(img.shape) == 3:
            ## this is an intensity mask, so we only apply to V channel
            new_img = np.ndarray(img.shape, dtype="complex64")
            Himg = img[:, :, 0]
            Simg = img[:, :, 1]
            Vimg = img[:, :, 2]
            new_img[:, :, 0] = Himg
            new_img[:, :, 1] = Simg
            new_img[:, :, 2] = np.multiply(Vimg, mask)
        elif len(img.shape) == 2:
            new_img = np.multiply(img, mask)
        return new_img
    
    ## H(u,v)
    def make_mask(img,H_func):
        N = img.shape[0]
        M = img.shape[1]
        mask = np.array([[H_func(i,j) for j in range(M)]for i in range(N)])
        return mask

    def ideal_low_pass(img, radius, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        h_func = lambda i,j: 1 if MaskFilter.d_to_center(N,M,i,j) < radius else 0
        mask = MaskFilter.make_mask(img,h_func)
        return MaskFilter.apply_mask(img, mask, filename)

    def ideal_high_pass(img, radius, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        h_func = lambda i,j: 1 if MaskFilter.d_to_center(N, M, i, j) >= radius else 0
        mask = MaskFilter.make_mask(img,h_func)
        return MaskFilter.apply_mask(img, mask, filename)

    def ideal_band_pass(img, r_high, r_low, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        mask = np.multiply(ideal_high_pass(r_low), ideal_low_pass(r_high))
        return MaskFilter.apply_mask(img, mask, filename)

    def butterworth_low_pass(img, radius, n, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        h_func = lambda i,j: 1 / (1 + (MaskFilter.d_to_center(N, M, i, j) / radius) ** (2 * n))
        mask = MaskFilter.make_mask(img,h_func)
        return MaskFilter.apply_mask(img, mask, filename)

    def butterworth_high_pass(img, radius, n, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        h_func = lambda i,j: 1/ (1+ (radius / (1 + MaskFilter.d_to_center(N, M, i, j))) ** (2 * n))
        mask = MaskFilter.make_mask(img,h_func)
        return MaskFilter.apply_mask(img, mask, filename)

    def butterworth_band_pass(img, r_high, r_low, r, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        mask = np.multiply(butterworth_high_pass(r_low, r), butterworth_low_pass(r_high, r))
        return MaskFilter.apply_mask(img, mask, filename)

    def gaussian_low_pass(img, radius, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        h_func = lambda i,j: e ** ((MaskFilter.d_to_center(N, M, i, j) ** 2) / (2 * radius ** 2))
        mask = MaskFilter.make_mask(img,h_func)
        return MaskFilter.apply_mask(img, mask, filename)

    def gaussian_high_pass(img, radius, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        h_func = lambda i,j: 1- e** ((MaskFilter.d_to_center(N, M, i, j) ** 2) / (2 * radius ** 2))
        mask = MaskFilter.make_mask(img,h_func)
        return MaskFilter.apply_mask(img, mask, filename)

    def gaussian_band_pass(img, r_high, r_low, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        mask = np.multiply(gaussian_high_pass(r_low), gaussian_low_pass(r_high))
        return MaskFilter.apply_mask(img, mask, filename)

    def get_turbulence(img,k):
        N = img.shape[0]
        M = img.shape[1]
        mask = np.array([[e**(-k*(u*u+v*v)**(5/6)) for v in range(M)] for u in range(N)])
        if len(img.shape)==3:
            new_mask = np.ndarray(img.shape)
            new_mask[:,:,0] = mask
            new_mask[:,:,1] = mask
            new_mask[:,:,2] = mask
        else:
            new_mask = mask
        return new_mask
    