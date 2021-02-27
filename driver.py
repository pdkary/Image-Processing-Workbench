from src.histogram_equalizer import HistogramEqualizer
from src.histogram_matcher import HistogramMatcher
from src.MaskFilter import MaskFilter
from src.fourier import FourierTransform
from src.adaptive_median_filter import AdaptiveMedianFilter
from src.kernel_filter import KernelFilter
from src.image_transformer import ImageTransformer
from src.image_utils import ImageUtils
import numpy as np
import cv2

blur_kernel = (
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

if __name__ == "__main__":

    filename1 = "adf_test"

    img1 = cv2.imread("images/" + filename1 + ".jpg")
    it = ImageTransformer(img1)

    it.transform(FourierTransform.transform)\
        .transform(MaskFilter.butterworth_high_pass,10,5,filename1)\
        .transform(FourierTransform.inverse_transform,filename1)\
        .transform(KernelFilter.filter,blur_kernel,filename1+"_bhp")\
        .transform(ImageUtils.substract_from,img1,.8,filename1)\
        .transform(AdaptiveMedianFilter.filter, 10)\
        .write("images/"+filename1+"_transformed.jpg")