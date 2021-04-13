import cv2
import numpy as np
import time
from src.image_utils import ImageUtils
from src.intensity_transformer import IntensityTransformer

def map_to_view(x):
    if x < 0:
        return 0
    if x > 255:
        return 255
    return x

class ImageTransformer:
    def __init__(self, img,debug=False):
        self.img = img
        self.debug = debug

    def transform(self, func, *args, **kwargs):
        if self.debug:
            print("beginning {}".format(func.__name__))
            print("| shape before: "+str(self.img.shape))
            time_before = time.time()
        self.img = func(self.img, *args, **kwargs)
        if self.debug:
            print("| shape after: "+str(self.img.shape))
            time_after = time.time()
            print("| took {} seconds".format(np.round(time_after-time_before,3)))
        return self
    
    def add(self,img,k=1):
        print("beginning add")
        img = np.uint8(img)
        if len(self.img.shape)==3:
            if len(img.shape)==3:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                self.img[:,:,2] += np.uint8(k*img[:,:,2])
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2RGB)

            if len(img.shape)==2:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                self.img[:,:,2] += np.uint8(k*img)
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2RGB)
        if len(self.img.shape)==2:
            if len(img.shape)==2:
                self.img += np.uint8(k*img)
            if len(img.shape)==3:
                raise ValueError("added image must have dimension <= src")
        self.img = np.vectorize(map_to_view)(self.img)
        return self

    def add_to(self,img,k=1):
        tmp_img = self.img
        self.img = img
        return self.add(tmp_img,k=k) 

    ##mask should be binary stepped ie) (0 || 255)
    def add_at(self,mask,img2,k=1,filename=None):
        bmask = mask/255
        anti_mask = 1-bmask
        new_img = np.ndarray(shape=self.img.shape)
        if(len(self.img.shape)==3):
            self.img[:,:,0] = np.multiply(anti_mask,self.img[:,:,0])
            self.img[:,:,1] = np.multiply(anti_mask,self.img[:,:,1])
            self.img[:,:,2] = np.multiply(anti_mask,self.img[:,:,2])
            new_img[:,:,0] = np.multiply(bmask,img2[:,:,0])
            new_img[:,:,1] = np.multiply(bmask,img2[:,:,1])
            new_img[:,:,2] = np.multiply(bmask,img2[:,:,2])
        elif len(self.img.shape)==2:
            self.img = np.multiply(anti_mask,self.img)
            new_img = np.multiply(bmask,img2)

        if filename is not None:
            cv2.imwrite("images/arithmetic/"+filename+"_add_by_mask.png",new_img)
        
        return self.add(new_img,k=k)
        
        
    
    def subtract(self,img,k=1):
        print("beginning subtract")
        img = np.uint8(img)
        if len(self.img.shape)==3:
            if len(img.shape)==3:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                self.img[:,:,2] -= np.uint8(k*img[:,:,2])
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2RGB)
            if len(img.shape)==2:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                self.img[:,:,2] -= np.uint8(k*img)
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2RGB)
        if len(self.img.shape)==2:
            if len(img.shape)==2:
                self.img -= np.uint8(k*img)
            if len(img.shape)==3:
                raise ValueError("subtracted image must have dimension <= src")
        self.img = np.vectorize(map_to_view)(self.img)
        return self
    
    def subtract_from(self,img,k=1):
        tmp_img = self.img
        self.img = img
        return self.subtract(tmp_img,k=k)
    
    def average(self,img,k=1):
        print("beginning average")
        if len(self.img.shape)==3:
            if len(img.shape)==3:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                self.img[:,:,2] = np.uint8((self.img[:,:,2] + k*img[:,:,2])/(1+k))
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2RGB)
            if len(img.shape)==2:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                self.img[:,:,2] = np.uint8((self.img[:,:,2]+k*img)/(k+1))
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2RGB)
        if len(self.img.shape)==2:
            if len(img.shape)==2:
                self.img = np.uint8((self.img + k*img)/(1+k))
            if len(img.shape)==3:
                raise ValueError("averaged image must have dimension <= src")
        self.img = np.vectorize(map_to_view)(self.img)
        return self

    def get(self):
        return self.img
        
    def write(self, filename):
        cv2.imwrite(filename, self.img)
        return self.img