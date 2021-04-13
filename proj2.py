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

    for i in range(9,19):
        if(i==9):
            filename_real = "mlts09_input"
            filename_synthetic = "kodim09_input"
        else:
            filename_real = "mlts"+str(i)+"_input"
            filename_synthetic = "kodim"+str(i)+"_input"
        
        img_real = cv2.imread("proj2_images/real/"+filename_real+".png")
        img_synth = cv2.imread("proj2_images/synthetic/"+filename_synthetic+".png")

        ImageTransformer(img_real,debug=True)\
            .transform(ScratchRemover.remove_vertical_small,filename=filename_real)\
                .write("proj2_images/out/"+filename_real+".png")

        ImageTransformer(img_real,debug=True)\
            .transform(ScratchRemover.remove_vertical_large,filename=filename_synthetic)\
                .write("proj2_images/out/"+filename_synthetic+".png")
