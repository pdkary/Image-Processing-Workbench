from src.image_transformer import ImageTransformer
from src.scratch_detector import ScratchDetector
from src.morphology import Morphology
from src.morphological_mask_maker import MorphologicalMaskMaker
from src.mask_blur import MaskBlur
from src.filter import Filter
from src.adaptive_median_filter import AdaptiveMedianFilter
from src.kernels import *
from src.convolution import Convolution
from src.mask_filter import MaskFilter

class ScratchRemover:


    @staticmethod
    def remove_vertical_large(img1,filename=None):
        scratches = ImageTransformer(img1,debug=True)\
                        .transform(ScratchDetector.detect_vertical_using_cross)\
                            .write("proj2_images/out/"+filename+"_scratches.png")

        expanded_scratches = ImageTransformer(scratches,debug=True)\
                                .transform(Morphology.dilate,MorphologicalMaskMaker.rectangle(3,3))\
                                    .write("proj2_images/out/"+filename+"_expanded_scratches.png")

        F_blurred = ImageTransformer(img1,debug=True)\
                        .transform(MaskBlur.blur_at,scratches,blur_x_strip_with_hole(25,1,7))\
                            .transform(Filter.max_filter,3,3)\
                                .transform(Filter.min_filter,3,3)\
                                    .write("proj2_images/out/"+filename+"_f_blurred.png")

        return ImageTransformer(img1,debug=True)\
                    .add_at(expanded_scratches,F_blurred)\
                        .transform(AdaptiveMedianFilter.filter_at,expanded_scratches,51)\
                            .transform(MaskBlur.blur_at,expanded_scratches,unsharpen_kernel_3)\
                                .get()

    @staticmethod
    def remove_vertical_small(img,filename=None):
        theta = np.pi/3
        thickness = 65
        blurred_diag_mask1 = Convolution.convolve(MaskFilter.diagonal_line(img,theta,thickness,block=False),blur_kernel(2*thickness+1))
        blurred_diag_mask2 = Convolution.convolve(MaskFilter.diagonal_line(img,np.pi-theta,thickness,block=False),blur_kernel(2*thickness+1))
        blurred_diag_mask = (blurred_diag_mask1+blurred_diag_mask2)/2
        
        return ImageTransformer(img,debug=True)\
                .transform(Convolution.convolve,blur_kernel(3))\
                    .transform(Convolution.convolve,blurred_diag_mask,convolve_g=False,filename=filename)\
                        .transform(Convolution.convolve,sharpen_kernel_3)\
                            .transform(Convolution.convolve,sharpen_kernel_3)\
                                .get()
        
