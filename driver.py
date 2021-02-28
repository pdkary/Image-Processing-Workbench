from src.histogram_equalizer import HistogramEqualizer
from src.histogram_matcher import HistogramMatcher
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

    filename1 = "20120_00_30s"

    img1 = cv2.imread("images/" + filename1 + ".jpg")
    turbulence = .000001
    ImageTransformer(img1,debug=True)\
        .transform(Filter.least_square_filter,gamma=.01,k=turbulence)\
            .write("images/"+filename1+"_ls.jpg")

    ImageTransformer(img1,debug=True)\
        .transform(Filter.wiener_filter,kt=turbulence,ks=1)\
            .write("images/"+filename1+"_wiener.jpg")

