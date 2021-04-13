from src.edge_detector import EdgeDetector
from src.morphology import Morphology
from src.morphological_mask_maker import MorphologicalMaskMaker
from src.image_transformer import ImageTransformer
from src.intensity_transformer import IntensityTransformer
from src.image_utils import ImageUtils
from src.convolution import Convolution
from src.mask_filter import MaskFilter
from src.kernels import *
import numpy as np

class ScratchDetector:

    @staticmethod
    def detect_vertical_sobel(img):
        vert_mask_11 = MorphologicalMaskMaker.rectangle(0,5)
        vert_mask_21 = MorphologicalMaskMaker.rectangle(0,10)
        square_mask_5 = MorphologicalMaskMaker.rectangle(2,2)
        
        return ImageTransformer(img,debug=True)\
                .transform(ImageUtils.convert_to_grayscale)\
                    .transform(EdgeDetector.sobel_x)\
                        .transform(IntensityTransformer.binary_step,k=100)\
                            .transform(Morphology.erode,vert_mask_21)\
                                .transform(Morphology.dilate,vert_mask_21)\
                                    .transform(Morphology.dilate,square_mask_5)\
                                        .get()
    
    @staticmethod
    def detect_vertical_high_pass(img,min_scrath_width,min_scratch_height):
        N = img.shape[0]
        M = img.shape[1]
        R = np.sqrt(N*N+M*M)/3
        vert_mask = MorphologicalMaskMaker.rectangle(0,min_scratch_height)
        square_mask = MorphologicalMaskMaker.rectangle(min_scrath_width,min_scrath_width)

        return ImageTransformer(img,debug=True)\
                .transform(ImageUtils.convert_to_grayscale)\
                    .transform(Convolution.convolve,MaskFilter.butterworth_high_pass(img,R,2),convolve_g=False)\
                        .transform(IntensityTransformer.binary_step,k=20)\
                            .transform(Morphology.open,square_mask)\
                                .transform(Convolution.convolve,blur_kernel(7))\
                                    .transform(IntensityTransformer.binary_step,k=128)\
                                        .transform(Morphology.open,vert_mask)\
                                            .get()
    @staticmethod
    def detect_vertical_using_cross(img):
        square_mask = MorphologicalMaskMaker.rectangle(1,1)
        big_rectangle_mask = MorphologicalMaskMaker.rectangle(2,8)
        return ImageTransformer(img,debug=True)\
                .transform(ImageUtils.convert_to_grayscale)\
                    .transform(Convolution.convolve,MaskFilter.inverse_cross(img,0,0,3),convolve_g=False)\
                        .transform(IntensityTransformer.map_intensities_to_viewable)\
                            .transform(EdgeDetector.sobel_x)\
                                .transform(IntensityTransformer.binary_step,128)\
                                    .transform(Morphology.open,square_mask)\
                                        .transform(Morphology.erode,big_rectangle_mask)\
                                            .get()