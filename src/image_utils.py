import numpy as np
import cv2


class ImageUtils:
    @staticmethod
    def add_border(img, b):
        return np.pad(img, b, mode="constant")
    
    @staticmethod
    def pad_to_size(img,new_shape):
        N = img.shape[0]
        M = img.shape[1]
        dx = (new_shape[0]-N)//2
        dy = (new_shape[1]-M)//2
        if len(img.shape)==3:
            new_shape = (new_shape[0],new_shape[1],img.shape[2])
            new_img = np.zeros(shape=new_shape,dtype=img.dtype)
            new_img[dx:N+dx,dy:M+dy,:]=img
        else:
            new_img = np.zeros(shape=new_shape,dtype=img.dtype)
            new_img[dx:N+dx,dy:M+dy]=img
        return new_img
    
    @staticmethod
    def add_border_color(img,t,r,b,l):
        N = img.shape[0]
        M = img.shape[1]
        new_img = np.zeros(shape=(N+r+l,M+t+b,3),dtype=img.dtype)
        new_img[r:N+r,t:M+t,:]=img
        return new_img

    @staticmethod
    def convert_to_grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def add(img1, img2,k=1,filename=None):
        new_img = img1 + k*img2
        if filename is not None:
            cv2.imwrite("images/arithmetic/"+filename+"_added.jpg",new_img)
        return new_img
    
    @staticmethod
    def add_to(img1,img2,k=1,filename=None):
        return ImageUtils.add(img2,img1,k=k,filename=filename)
    
    @staticmethod
    def average(img1,img2,k=1,filename=None):
        new_img = (img1+k*img2)/2
        if filename is not None:
            cv2.imwrite("images/arithmetic/"+filename+"_averaged.jpg",new_img)
        return new_img

    @staticmethod
    def subtract(img1, img2,k=1,filename=None):
        new_img = img1 - k*img2
        if filename is not None:
            cv2.imwrite("images/arithmetic/"+filename+"_subtracted.jpg",new_img)
        return new_img

    @staticmethod
    def subtract_from(img1, img2,k=1,filename=None):
        new_img = img2 - k*img1
        if filename is not None:
            cv2.imwrite("images/arithmetic/"+filename+"_subtracted.jpg",new_img)
        return new_img
    
    @staticmethod
    def map_intensities_to_viewable(img):
        max_val = np.max(img)
        min_val = np.min(img)
        f = lambda x: 255*(x-min_val)/(max_val-min_val)
        return np.vectorize(f)(img)
