import numpy as np
import cv2


class ImageUtils:
    @staticmethod
    def add_border(img, b):
        return np.pad(img, b, mode="constant")

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
    def average(img1,img2,filename=None):
        new_img = (img1+img2)/2
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
