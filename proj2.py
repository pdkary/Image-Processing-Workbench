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
from src.morphology import Morphology
from src.morphological_mask_maker import MorphologicalMaskMaker
from src.edge_detector import EdgeDetector
from src.convolution import Convolution
from src.scratch_detector import ScratchDetector
from src.filter import Filter
from src.kernels import *
from src.mask_blur import MaskBlur
import numpy as np
import cv2

if __name__ == "__main__":

    #IMAGE 1
    filename = "kodim09_input"
    img1 = cv2.imread("proj2_images/synthetic/"+filename+".png")

    img1_mask = cv2.imread("proj2_images/out/"+filename+"_mask.png")

    scratches = ImageTransformer(img1_mask,debug=True)\
                    .transform(ImageUtils.convert_to_grayscale)\
                        .get()

    # scratches = ImageTransformer(img1,debug=True)\
    #                 .transform(ScratchDetector.detect_vertical_high_pass,1,10)\
    #                     .write("proj2_images/out/"+filename+"_mask.png")\
    # scratches = ImageTransformer(img1,debug=True)\
    #                 .transform(ScratchDetector.detect_vertical_using_cross)\
    #                     .write("proj2_images/out/"+filename+"_mask.png")

    F_blurred = ImageTransformer(img1,debug=True)\
                    .transform(MaskBlur.blur_at,scratches,blur_x_strip_with_hole(21,3,7))\
                        .transform(AdaptiveMedianFilter.filter,51)\
                        .write("proj2_images/out/"+filename+".png")

                        # .transform(AdaptiveMedianFilter.filter_at,scratches,100)\
    # ImageTransformer(F_scratched-F_blurred,debug=True)\
    #     .transform(IntensityTransformer.map_intensities_to_viewable)\
    #         .transform(np.abs)\
    #     .write("proj2_images/out/"+filename+"_fourier.png")
    

                    # .write("proj2_images/out/"+filename+".png")
        # .transform(MaskBlur.blur_at,scratches,blur_x_strip_with_hole(31,5,11))\
            # .transform(Convolution.convolve,sharpen_kernel_3)\
                    # .transform(AdaptiveMedianFilter.filter_at,scratches,51)\
                        # .transform(Convolution.convolve, MaskFilter.butterworth_low_pass(img1,400,1),False,'test',)\
                # .transform(AdaptiveMedianFilter.filter_at,scratches,101)\
                # .transform(MaskBlur.blur_at,scratches,blur_x_strip(31,5))\
                