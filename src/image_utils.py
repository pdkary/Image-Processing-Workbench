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

