from src.histogram_equalizer import HistogramEqualizer
from src.histogram_matcher import HistogramMatcher
from src.homomorphic_filter import HomomorphicFilter
from src.hough_transformer import HoughTransformer
from src.mask_filter import MaskFilter
from src.fourier import FourierTransform
from src.adaptive_median_filter import AdaptiveMedianFilter
from src.image_transformer import ImageTransformer
from src.image_utils import ImageUtils
from src.intensity_transformer import IntensityTransformer
from src.edge_detector import EdgeDetector
from src.convolution import Convolution
from src.filter import Filter
from src.kernels import *
import numpy as np
import cv2

if __name__ == "__main__":

    #IMAGE 1
    # img1 = cv2.imread("images/duck.jpg")
    img1 = cv2.imread("images/duck.jpg")
    img_mag = cv2.imread("images/hough/triangle_mag.jpg")
    # img2 = cv2.imread("images/horiz_bars.jpg")
    # img3 = cv2.imread("images/diag_bars.jpg")
    # img4 = cv2.imread("images/hypno.jpg")
    # img5 = cv2.imread("images/fingerprint.jpg")
    # img6 = cv2.imread("images/seeds.jpg")
    # img7 = cv2.imread("images/checker.jpg")

    ImageTransformer(img1,debug=True)\
        .transform(HoughTransformer.transform,"triangle")\
            .write("images/hough/triangle.jpg")