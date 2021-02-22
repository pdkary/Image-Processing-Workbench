import cv2
import numpy as np

class ColorSplitter:
    def __init__(self,img,filename):
        self.img = img
        self.filename = filename
        self.N = self.img.shape[0]
        self.M = self.img.shape[1]

    def split(self):
        self.R_img = self.img[:,:,0]
        self.B_img = self.img[:,:,1]
        self.G_img = self.img[:,:,2]
        return [self.R_img,self.G_img,self.B_img]
    
    def combine(self,R_img,G_img,B_img):
        combined_img = np.ndarray(self.img.shape)
        combined_img[:,:,0]=R_img
        combined_img[:,:,1]=G_img
        combined_img[:,:,2]=B_img
        return combined_img
    
    def save(self):
        cv2.imwrite("images/split/"+self.filename+"_red.jpg",self.R_img)
        cv2.imwrite("images/split/"+self.filename+"_green.jpg",self.G_img)
        cv2.imwrite("images/split/"+self.filename+"_blue.jpg",self.B_img)


    