from src.histogram_equalizer import HistogramEqualizer
from src.histogram_matcher import HistogramMatcher
from src.homomorphic_filter import HomomorphicFilter
from src.mask_filter import MaskFilter
from src.fourier import FourierTransform
from src.adaptive_median_filter import AdaptiveMedianFilter
from src.image_transformer import ImageTransformer
from src.image_utils import ImageUtils
from src.intensity_transformer import IntensityTransformer
from src.convolution import Convolution
from src.filter import Filter
from src.kernels import *
import numpy as np
import cv2

if __name__ == "__main__":

    #IMAGE 1
    img1 = cv2.imread("images/stanford.jpg")
    img2 = cv2.imread("images/horiz_bars.jpg")
    img3 = cv2.imread("images/diag_bars.jpg")
    img4 = cv2.imread("images/hypno.jpg")
    img5 = cv2.imread("images/fingerprint.jpg")
    img6 = cv2.imread("images/seeds.jpg")
    img7 = cv2.imread("images/checker.jpg")

    N = img1.shape[0]
    M = img1.shape[1]
    
    d_to_center = lambda x,y: np.sqrt((x - N / 2) ** 2 + (y - M / 2) ** 2)
    #butterworth low pass
    radius = 10
    n = 10
    butter_low = lambda i,j: 1 / (1 + (d_to_center(i, j) / radius) ** (2 * n))
    butter_high = lambda i,j: 1/ (1+ (radius / (1 + d_to_center(i, j))) ** (2 * n))
    
    ImageTransformer(img1)\
        .transform(HomomorphicFilter.filter,butter_high,filename="fuck")\
            .write("images/filtered/stanford.jpg")
    
    ImageTransformer(img2)\
        .transform(ImageUtils.convert_to_grayscale)\
        .transform(FourierTransform.transform)\
            .transform(FourierTransform.get_viewable_fourier)\
                .write("images/fourier/horiz_bars.jpg")
    
    ImageTransformer(img3)\
        .transform(ImageUtils.convert_to_grayscale)\
        .transform(FourierTransform.transform)\
            .transform(FourierTransform.get_viewable_fourier)\
                .write("images/fourier/diag_bars.jpg")

    ImageTransformer(img4)\
        .transform(ImageUtils.convert_to_grayscale)\
        .transform(FourierTransform.transform)\
            .transform(FourierTransform.get_viewable_fourier)\
                .write("images/fourier/hypno.jpg")

    ImageTransformer(img5)\
        .transform(ImageUtils.convert_to_grayscale)\
        .transform(FourierTransform.transform)\
            .transform(FourierTransform.get_viewable_fourier)\
                .write("images/fourier/fingerprint.jpg")
    
    ImageTransformer(img6)\
        .transform(ImageUtils.convert_to_grayscale)\
        .transform(FourierTransform.transform)\
            .transform(FourierTransform.get_viewable_fourier)\
                .write("images/fourier/seeds.jpg")

    ImageTransformer(img7)\
        .transform(ImageUtils.convert_to_grayscale)\
        .transform(FourierTransform.transform)\
            .transform(FourierTransform.get_viewable_fourier)\
                .write("images/fourier/checker.jpg")
