from src.histogram_equalizer import HistogramEqualizer
from src.histogram_matcher import HistogramMatcher
from src.MaskFilter import MaskFilter
from src.fourier import FourierTransform
from src.adaptive_median_filter import AdaptiveMedianFilter
from src.kernel_filter import KernelFilter
from src.image_transformer import ImageTransformer
from src.image_utils import ImageUtils
from src.intensity_transformer import IntensityTransformer
import numpy as np
import cv2

blur_kernel_5 = (
    np.array(
        [
            [1, 2, 4, 2, 1],
            [2, 4, 8, 4, 2],
            [4, 8, 16, 8, 4],
            [2, 4, 8, 4, 2],
            [1, 2, 4, 2, 1],
        ]
    )
    / 100
)

blur_kernel_3 = (
    np.array(
        [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ]
    )
    / 16
)

if __name__ == "__main__":

    filename1 = "adf_test"

    img1 = cv2.imread("images/" + filename1 + ".jpg")
    it = ImageTransformer(img1)

    it.transform(FourierTransform.transform)\
        .transform(MaskFilter.butterworth_high_pass,20,10,filename1)\
        .transform(FourierTransform.inverse_transform,filename1)\
        .transform(ImageUtils.subtract_from,img1,.5,filename1)\
        .transform(AdaptiveMedianFilter.filter, 50)\
        .write("images/"+filename1+"_transformed.jpg")