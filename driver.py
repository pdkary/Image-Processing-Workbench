from src.histogram_equalizer import HistogramEqualizer
from src.histogram_matcher import HistogramMatcher
from src.MaskFilter import MaskFilter
from src.fourier import FourierTransform
from src.adaptive_median_filter import AdaptiveMedianFilter
from src.image_transformer import ImageTransformer
from src.image_utils import ImageUtils
from src.intensity_transformer import IntensityTransformer
from src.convolution import Convolution
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
laplace_kernel_3 = (
    np.array(
        [
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1],
        ]
    )
)
if __name__ == "__main__":

    filename1 = "solar_system"
    filename2 = "saturn"

    img1 = cv2.imread("images/" + filename1 + ".jpg")
    img2 = cv2.imread("images/" + filename2 + ".jpg")

    g_img2 = ImageUtils.convert_to_grayscale(img2)

    ImageTransformer(img1,debug=True)\
        .transform(Convolution.convolve,laplace_kernel_3)\
            .transform(ImageUtils.add_to,img1,k=.5)\
            .write("images/convolution/"+filename1+"_post.jpg")
