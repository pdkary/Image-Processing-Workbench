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
from src.scratch_remover import ScratchRemover
import numpy as np
import cv2

if __name__ == "__main__":

    filename = "mlts09_input"
    img = cv2.imread("proj2_images/real/"+filename+".png")

    ImageTransformer(img,debug=True)\
        .transform(ScratchRemover.remove_vertical_small,filename=filename)\
            .write("proj2_images/out/"+filename+".png")
