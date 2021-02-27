from src.image_utils import ImageUtils
import numpy as np
import cv2

class KernelFilter:
    @staticmethod
    def filter(img,kernel,filename=None):
        new_img = np.ndarray(img.shape)
        if len(img.shape)==3:
            new_img = KernelFilter.filter_color(img,kernel)
        if len(img.shape)==2:
            new_img =  KernelFilter.filter_flat(img,kernel)
        
        if filename is not None:
            cv2.imwrite("images/filtered/"+filename+".jpg",new_img)
        return new_img

    @staticmethod
    def filter_color(img, kernel):
        new_img = np.ndarray(shape=img.shape)
        new_img[:,:,0] = KernelFilter.filter_flat(img[:,:,0],kernel)
        new_img[:,:,1] = KernelFilter.filter_flat(img[:,:,1],kernel)
        new_img[:,:,2] = KernelFilter.filter_flat(img[:,:,2],kernel)
        return new_img
    
    @staticmethod
    def filter_flat(img,kernel):
        N = img.shape[0]
        M = img.shape[1]
        kw = kernel.shape[0] - 1
        r = kw // 2
        b_img = np.ndarray(shape=(img.shape[0] + kw, img.shape[1] + kw))
        b_img = ImageUtils.add_border(img,r)
        new_img = np.ndarray(shape=img.shape)

        for i in range(r, N + r - 1):
            for j in range(r, M + r - 1):
                new_val = KernelFilter.process_window(b_img, i, j, kernel)
                new_img[i - r, j - r] = new_val
        return new_img
    
    @staticmethod
    def process_window(img,i,j,kernel):
        r = (kernel.shape[0]-1)//2
        window = img[i-r:i+r+1,j-r:j+r+1]
        val = int(np.sum(np.multiply(kernel,window)))
        return val
