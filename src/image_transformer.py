import cv2
import numpy as np

class ImageTransformer:
    def __init__(self, img,debug=False):
        self.img = img
        self.debug = debug

    def transform(self, func, *args, **kwargs):
        if self.debug:
            print("beginning {}".format(func.__name__))
            print("| shape before: "+str(self.img.shape))
        self.img = func(self.img, *args, **kwargs)
        if self.debug:
            print("| shape after: "+str(self.img.shape))
        return self
    
    def add(self,img,k=1):
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
        return self

    def add_to(self,img,k=1):
        tmp_img = self.img
        self.img = img
        return self.add(tmp_img,k=k)   
    
    def subtract(self,img,k=1):
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
        return self
    
    def subtract_from(self,img,k=1):
        tmp_img = self.img
        self.img = img
        return self.subtract(tmp_img,k=k)
    
    def average(self,img,k):
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
        return self

    def get(self):
        return self.img
        
    def write(self, filename):
        cv2.imwrite(filename, self.img)
