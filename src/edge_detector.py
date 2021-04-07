import numpy as np
from src.convolution import Convolution
from src.kernels import sobel_horizontal,sobel_vertical,blur_kernel
blur_kernel_5 = blur_kernel(5)
class EdgeDetector:
    @staticmethod
    def sobel(img,return_edges=False):
        Gx = Convolution.convolve(img,sobel_horizontal)
        Gy = Convolution.convolve(img,sobel_vertical)
        mag =  np.sqrt(np.multiply(Gx,Gx)+np.multiply(Gy,Gy))
        if return_edges:
            return Gx, Gy, mag
        else:
            return mag
    
    @staticmethod
    def sobel_with_blur(img,return_edges=False):
        nGx,nGy,nMag = EdgeDetector.sobel(img,True)

        bnGx = Convolution.convolve(nGx,blur_kernel_5)
        bnGy = Convolution.convolve(nGy,blur_kernel_5)
        bnMag = Convolution.convolve(nMag,blur_kernel_5)
        
        blurred_img = Convolution.convolve(img,blur_kernel_5)
        bGx,bGy,bMag = EdgeDetector.sobel(blurred_img,True)
        
        oGx,oGy,oMag = (bnGx+bGx)/2,(bnGy+bGy)/2,(bnMag+bMag)/2

        if return_edges:
            return oGx,oGy,oMag
        else:
            return oMag
    
    @staticmethod
    def sobel_x(img):
        gx,gy,gMag = EdgeDetector.sobel(img,return_edges=True)
        return gx
    
    @staticmethod
    def sobel_y(img):
        gx,gy,gMag = EdgeDetector.sobel(img,return_edges=True)
        return gy
    
    @staticmethod
    def sobel_x_with_blur(img):
        gx,gy,gMag = EdgeDetector.sobel_with_blur(img,return_edges=True)
        return gx
    
    @staticmethod
    def sobel_y_with_blur(img):
        gx,gy,gMag = EdgeDetector.sobel_with_blur(img,return_edges=True)
        return gy
