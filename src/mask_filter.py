import numpy as np
from math import e, pi
import cv2
from src.fourier import FourierTransform


class MaskFilter:
    def d_to_center(N, M, x, y):
        return np.sqrt((x - N / 2) ** 2 + (y - M / 2) ** 2)

    def apply_mask(img, mask, filename=None):
        if len(img.shape) == 3:
            new_img = np.ndarray(img.shape, dtype="complex64")
            Rimg = img[:, :, 0]
            Gimg = img[:, :, 1]
            Bimg = img[:, :, 2]
            new_img[:, :, 0] = np.multiply(Rimg, mask)
            new_img[:, :, 1] = np.multiply(Gimg, mask)
            new_img[:, :, 2] = np.multiply(Bimg, mask)
        elif len(img.shape) == 2:
            new_img = np.multiply(img, mask)

        if filename is not None:
            cv2.imwrite(
                "images/fourier/filtered/" + filename + ".jpg",
                FourierTransform.get_viewable_fourier(new_img),
            )
        return new_img

    def ideal_low_pass(img, radius, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        mask = np.array(
            [
                [
                    1 if MaskFilter.d_to_center(N, M, i, j) < radius else 0
                    for j in range(M)
                ]
                for i in range(N)
            ]
        )
        return MaskFilter.apply_mask(img, mask, filename)

    def ideal_high_pass(img, radius, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        mask = np.array(
            [
                [
                    1 if MaskFilter.d_to_center(N, M, i, j) >= radius else 0
                    for j in range(M)
                ]
                for i in range(N)
            ]
        )
        return MaskFilter.apply_mask(img, mask, filename)

    def ideal_band_pass(img, r_high, r_low, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        mask = np.multiply(ideal_high_pass(r_low), ideal_low_pass(r_high))
        return MaskFilter.apply_mask(img, mask, filename)

    def butterworth_low_pass(img, radius, n, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        mask = np.array(
            [
                [
                    1 / (1 + (MaskFilter.d_to_center(N, M, i, j) / radius) ** (2 * n))
                    for j in range(M)
                ]
                for i in range(N)
            ]
        )
        return MaskFilter.apply_mask(img, mask, filename)

    def butterworth_high_pass(img, radius, n, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        mask = np.array(
            [
                [
                    1
                    / (
                        1
                        + (radius / (1 + MaskFilter.d_to_center(N, M, i, j))) ** (2 * n)
                    )
                    for j in range(M)
                ]
                for i in range(N)
            ]
        )
        return MaskFilter.apply_mask(img, mask, filename)

    def butterworth_band_pass(img, r_high, r_low, r, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        mask = np.multiply(
            butterworth_high_pass(r_low, r), butterworth_low_pass(r_high, r)
        )
        return MaskFilter.apply_mask(img, mask, filename)

    def gaussian_low_pass(img, radius, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        mask = np.array(
            [
                [
                    e ** ((MaskFilter.d_to_center(N, M, i, j) ** 2) / (2 * radius ** 2))
                    for j in range(M)
                ]
                for i in range(N)
            ]
        )
        return MaskFilter.apply_mask(img, mask, filename)

    def gaussian_high_pass(img, radius, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        mask = np.array(
            [
                [
                    1
                    - e
                    ** ((MaskFilter.d_to_center(N, M, i, j) ** 2) / (2 * radius ** 2))
                    for j in range(M)
                ]
                for i in range(N)
            ]
        )
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
    