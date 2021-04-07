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

    # #IMAGE 1
    # img1 = cv2.imread("images/10087_00_30s.jpg")
    # img2 = cv2.imread("images/20107_00_30s.jpg")
    
    # ImageTransformer(img1,debug=True)\
    #     .transform(HistogramMatcher.match,img2,filenames=("10087_00_30s","20107_00_30s"))\
    #             .transform(IntensityTransformer.map_intensities_to_viewable)\
    #                 .write("images/10087_00_30s_FINAL.jpg")
    #IMAGE 2
    img1 = cv2.imread("images/test4.png")
    ImageTransformer(img1,debug=True)\
        .transform(IntensityTransformer.upper_threshold,k=150)\
            .subtract_from(img1,k=.45)\
                .transform(IntensityTransformer.map_intensities_to_viewable)\
                    .write("images/test4_t.jpg")
    
    # #IMAGE 3
    # img1 = cv2.imread("images/20120_00_30s.jpg")
    # ImageTransformer(img1,debug=True)\
    #     .transform(IntensityTransformer.upper_threshold,k=150,filename="20120_00_30s_thresholded")\
    #         .subtract_from(img1,k=.45)\
    #             .transform(IntensityTransformer.map_intensities_to_viewable)\
    #                 .write("images/20120_00_30s_t.jpg")
    
    # #IMAGE 4
    # img1 = cv2.imread("images/20147_00_30s.jpg")
    # img2 = cv2.imread("images/stanford.jpg")
    # ImageTransformer(img1,debug=True)\
    #     .transform(HistogramMatcher.match,img2)\
    #         .write("images/20147_00_30s_t.jpg")

