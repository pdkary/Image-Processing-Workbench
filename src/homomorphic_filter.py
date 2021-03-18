import numpy as np
from src.image_utils import ImageUtils
from src.fourier import FourierTransform
from src.mask_filter import MaskFilter
from src.kernels import *
import cv2

class HomomorphicFilter:
    """
    filter func must take arguments (img,radius)
    """
    @staticmethod
    def filter(img,H_func,filename=None):
        

        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        log_img = np.log(img)

        fourier_img = FourierTransform.transform(log_img,filename=filename)
        
        mask = MaskFilter.make_mask(fourier_img,H_func)
        
        filtered_fourier_img = MaskFilter.apply_mask(fourier_img,mask,filename=filename)

        recovered_img = FourierTransform.inverse_transform(filtered_fourier_img,filename=filename)

        unlogged_img = np.uint8(np.exp(recovered_img))
        
        filtered_img = cv2.cvtColor(unlogged_img,cv2.COLOR_HSV2RGB)
        return filtered_img



