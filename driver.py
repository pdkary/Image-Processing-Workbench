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

    filename1 = "10087_00_30s"

    img1 = cv2.imread("images/"+filename1+".jpg")
    
    med_light_img = ImageTransformer(img1).transform(IntensityTransformer.shift,10).get()
    high_light_img = ImageTransformer(img1).transform(IntensityTransformer.shift,20).get()

    ImageTransformer(img1)\
        .transform(HistogramEqualizer.equalize)\
            .transform(IntensityTransformer.linear_cutoff,250)\
                .write("images/"+filename1+"_t.jpg")
    # img1 = cv2.imread("images/20147_00_30s.jpg")
    # img2 = cv2.imread("images/stanford.jpg")
    # ImageTransformer(img1,debug=True)\
    #     .transform(HistogramMatcher.match,img2)\
    #         .write("images/"+filename1+"_t.jpg")

