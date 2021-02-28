from src.filter_base import FilterBase
from src.mask_filter import MaskFilter
from src.fourier import FourierTransform
from src.image_utils import ImageUtils
from src.kernels import laplace_no_diag_3
import numpy as np

class Filter(FilterBase):
    contraharmonic_Q = 1.5
    @staticmethod
    def filter_arithmetic_mean(img,n,m,filename=None):
        return Filter.filter(Filter.process_window_arithmetic_mean,img,n,m,filename)
    
    @staticmethod
    def filter_geometric_mean(img,n,m,filename=None):
        return Filter.filter(Filter.process_window_geometric_mean,img,n,m,filename)

    @staticmethod
    def filter_harmonic_mean(img,n,m,filename=None):
        return Filter.filter(Filter.process_window_harmonic_mea,n,m,filename)

    @staticmethod
    def filter_contraharmonic_mean(img,n,m,Q,filename=None):
        Filter.contraharmonic_Q = Q
        return Filter.filter(Filter.process_window_contraharmonic_mean,img,n,m,filename)

    @staticmethod
    def process_window_arithmetic_mean(img,i,j,n,m):
        window = img[i-n//2:i+1+n//2,j-m//2:j+1+m//2]
        return window.sum()/(n*m)
    
    @staticmethod
    def process_window_geometric_mean(img,i,j,n,m):
        window = img[i-n//2+1:i+n//2,j-m//2+1:j+m//2]
        return window.prod()**(1/(n*m))
    
    @staticmethod
    def process_window_harmonic_mean(img,i,j,n,m):
        window = img[i-n//2+1:i+n//2,j-m//2+1:j+m//2]
        inverse = lambda x: 1/x
        window = np.vectorize(inverse)(window)
        return m*n/window.sum()

    @staticmethod
    def process_window_contraharmonic_mean(img,i,j,n,m):
        window = img[i-n//2:i+1+n//2,j-m//2:j+1+m//2]
        return (window.sum()**(Filter.contraharmonic_Q+1))/(window.sum()**(Filter.contraharmonic_Q))
    
    @staticmethod
    def wiener_filter(img,kt,ks):
        H = MaskFilter.get_turbulence(img,kt)
        f_img = FourierTransform.transform(img)
        G = H*f_img
        F_est = ((1/H)*(H*H)/(H*H+ks))*G

        F_est = FourierTransform.reshape_fourier(F_est)
        f_est = FourierTransform.inverse_transform(F_est)
        return f_est
    
    @staticmethod
    def least_square_filter(img,gamma,k):
        H = MaskFilter.get_turbulence(img,k)
        f_img = FourierTransform.transform(img)
        G = H*f_img
        P = FourierTransform.transform(ImageUtils.pad_to_size(laplace_no_diag_3,img.shape))
        
        F_est = (H/(H*H+gamma*P*P))*G
        F_est = FourierTransform.reshape_fourier(F_est)
        f_est = FourierTransform.inverse_transform(F_est)
        return f_est
        