import numpy as np
import cv2
from math import e, pi
from tqdm import tqdm


class MaskMaker:
    def __init__(self, N, M):
        self.N = N
        self.M = M

    def d_to_center(self, x, y):
        return np.sqrt((x-self.N/2)**2+(y-self.M/2)**2)

    def ideal_low_pass(self, radius):
        return np.array([[1 if self.d_to_center(i, j) < radius else 0
                          for j in range(self.M)] for i in range(self.N)])

    def ideal_high_pass(self, radius):
        return np.array([[1 if self.d_to_center(i, j) >= radius else 0
                          for j in range(self.M)] for i in range(self.N)])

    def ideal_band_pass(self,r_high,r_low):
        return np.multiply(self.ideal_high_pass(r_low),self.ideal_low_pass(r_high))

    def butterworth_low_pass(self, radius, n):
        return np.array([[1/(1+(self.d_to_center(i, j)/radius)**(2*n))
                          for j in range(self.M)]for i in range(self.N)])

    def butterworth_high_pass(self, radius, n):
        return np.array([[1/(1+(radius/(1+self.d_to_center(i, j)))**(2*n))
                          for j in range(self.M)]for i in range(self.N)])
    
    def butterworth_band_pass(self,r_high,r_low,r):
        return np.multiply(self.butterworth_high_pass(r_low,r),self.butterworth_low_pass(r_high,r))

    def gaussian_low_pass(self, radius):
        return np.array([[e**((self.d_to_center(i, j)**2)/(2*radius**2))
                          for j in range(self.M)] for i in range(self.N)])

    def gaussian_high_pass(self, radius):
        return np.array([[1-e**((self.d_to_center(i, j)**2)/(2*radius**2))
                          for j in range(self.M)] for i in range(self.N)])

    def gaussian_band_pass(self,r_high,r_low):
        return np.multiply(self.gaussian_high_pass(r_low),self.gaussian_low_pass(r_high))


class FourierTransform:
    def __init__(self, img):
        self.N = img.shape[0]
        self.M = img.shape[1]
        self.reshape_basis = np.array(
            [[(-1)**(i+j) for i in range(self.M)]for j in range(self.N)])
        self.img = self.reshape_img(img)
        self.fourier_img = np.array((self.N,self.M,1))

    def reshape_img(self, img):
        reshaped_img = np.multiply(img, self.reshape_basis)
        return reshaped_img

    def filter(self, mask):
        self.fourier_img = np.multiply(mask, self.fourier_img)
        return self.get_viewable_img()

    def transform_forward(self):
        self.fourier_img = np.fft.fft2(self.img)
        return self.get_viewable_img()

    def transform_backward(self):
        return np.abs(np.fft.ifft2(self.fourier_img))

    def get_viewable_img(self):
        fmax = np.amax(self.fourier_img)
        c = 255/np.log(1+np.abs(fmax))
        return c*np.log(1+np.abs(self.fourier_img))
