import numpy as np
from src.image_utils import ImageUtils

class MaskBlur:

    @staticmethod
    def blur_at(img,blur_mask,blur_kernel):
        if len(img.shape)==3:
            new_img = np.ndarray(shape=img.shape)
            new_img[:,:,0] = MaskBlur.blur_at_flat(img[:,:,0],blur_mask,blur_kernel)
            new_img[:,:,1] = MaskBlur.blur_at_flat(img[:,:,1],blur_mask,blur_kernel)
            new_img[:,:,2] = MaskBlur.blur_at_flat(img[:,:,2],blur_mask,blur_kernel)
            return new_img
        elif len(img.shape)==2:
            return MaskBlur.blur_at_flat(img,blur_mask,blur_kernel)

    @staticmethod
    def blur_at_flat(img,blur_mask,blur_kernel):
        N = img.shape[0]
        M = img.shape[1]
        if N != blur_mask.shape[0] or M != blur_mask.shape[1]:
            raise ValueError("image and mask must have the same shape")
        
        kN = blur_kernel.shape[0]
        kM = blur_kernel.shape[1]
        bN = kN//2
        bM = kM//2
        oN = kN%2
        oM = kM%2
        new_shape = (N+kN,M+kM)
        img = ImageUtils.pad_to_size(img,new_shape)
        blur_mask = ImageUtils.pad_to_size(blur_mask,new_shape)

        new_img = np.ndarray(shape=img.shape)

        for i in range(bN,N+bN):
            for j in range(bM,M+bM):
                if blur_mask[i,j]==255:
                    new_img[i,j] = MaskBlur.process_window(img,i,j,blur_kernel)
                else:
                    new_img[i,j] = img[i,j]
        return new_img[bN:N+bN,bM:M+bM]
    

    @staticmethod
    def process_window(img,i,j,kernel):
        sx = kernel.shape[0]//2
        sy = kernel.shape[1]//2
        ox = kernel.shape[0]%2
        oy = kernel.shape[1]%2

        window = img[i-sx+1-ox:i+sx+ox,j-sy+1-oy:j+sy+oy]
        out = np.multiply(window,kernel)
        return np.sum(out)

