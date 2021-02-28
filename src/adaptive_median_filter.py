from src.image_utils import ImageUtils
import numpy as np
import cv2

class AdaptiveMedianFilter:
    @staticmethod
    def filter(img, smax,filename=None):
        new_img = np.ndarray(img.shape)
        if len(img.shape) == 3:
            new_img = AdaptiveMedianFilter.filter_color(img, smax)
        elif len(img.shape) == 2:
            new_img = AdaptiveMedianFilter.filter_flat(img, smax)

        if filename is not None:
            cv2.imwrite("images/filtered/"+filename+"_amf.jpg",new_img)
        return new_img

    @staticmethod
    def filter_flat(img, smax):
        N = img.shape[0]
        M = img.shape[1]
        sxy = 1  # initial value
        b_img = ImageUtils.add_border(img=img, b=smax)
        new_img = np.ndarray(b_img.shape)
        for i in range(smax, N + smax):
            for j in range(smax, M + smax):
                new_val = AdaptiveMedianFilter.process_window(b_img, i, j, sxy, smax)
                new_img[i, j] = new_val

        return new_img[smax : N + smax, smax : M + smax]

    @staticmethod
    def filter_color(img, smax):
        new_img = np.ndarray(img.shape)
        Rimg = img[:, :, 0]
        Gimg = img[:, :, 1]
        Bimg = img[:, :, 2]
        new_img[:, :, 0] = AdaptiveMedianFilter.filter_flat(Rimg, smax)
        new_img[:, :, 1] = AdaptiveMedianFilter.filter_flat(Gimg, smax)
        new_img[:, :, 2] = AdaptiveMedianFilter.filter_flat(Bimg, smax)
        return new_img

    @staticmethod
    def process_window(img, i, j, sxy, smax):
        window = img[i - sxy + 1 : i + sxy, j - sxy + 1 : j + sxy]
        Zxy = img[i, j]

        Zmin, Zmed, Zmax = np.min(window), np.median(window), np.max(window)
        A1 = Zmed - Zmin
        A2 = Zmed - Zmax
        if A1 > 0 and A2 < 0:
            B1 = Zxy - Zmin
            B2 = int(Zxy) - int(Zmax)
            if B1 > 0 and B2 < 0:
                return Zxy
            else:
                return Zmed
        else:
            sxy += 1
            if sxy == smax:
                return Zxy
            else:
                return AdaptiveMedianFilter.process_window(img, i, j, sxy, smax)
