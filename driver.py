from src.histogram_equalizer import HistogramEqualizer
from src.histogram_matcher import HistogramMatcher
from src.MaskFilter import MaskFilter
from src.fourier import FourierTransform
from src.adaptive_median_filter import AdaptiveMedianFilter
from src.image_transformer import ImageTransformer
from src.image_utils import ImageUtils
from src.intensity_transformer import IntensityTransformer
from src.convolution import Convolution
from src.kernels import *
import numpy as np
import cv2

if __name__ == "__main__":

    filename1 = "amf"
    filename2 = "shape"

    img1 = cv2.imread("images/" + filename1 + ".jpg")
    img2 = cv2.imread("images/" + filename2 + ".jpg")
    
    g_img1 = ImageUtils.convert_to_grayscale(img1)
    g_img2 = ImageUtils.convert_to_grayscale(img2)

    ImageTransformer(img1,debug=True)\
        .transform(ImageUtils.convert_to_grayscale)\
            .transform(AdaptiveMedianFilter.filter,3)\
            .transform(AdaptiveMedianFilter.filter,10)\
            .transform(AdaptiveMedianFilter.filter,90)\
            .transform(AdaptiveMedianFilter.filter,250)\
                .write("images/"+filename1+"_transformed.jpg")


