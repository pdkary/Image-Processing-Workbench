import cv2
from color_splitter import ColorSplitter
from fourier import FourierTransform,MaskMaker
import numpy as np 

class ColorFourier:
    def __init__(self,filename,debug=True):
        self.filename = filename
        self.img = cv2.imread("images/"+filename+".jpg")
        self.N = self.img.shape[0]
        self.M = self.img.shape[1]
        self.mask_maker = MaskMaker(self.N,self.M)
        self.debug = debug
        self.cs = ColorSplitter(self.img,self.filename)
        self.R_img,self.B_img,self.G_img = self.cs.split()
        self.cs.save()

    def filter(self,mask):
        R_fourier = FourierTransform(self.R_img)
        G_fourier = FourierTransform(self.G_img)
        B_fourier = FourierTransform(self.B_img)

        R_fourier_img = R_fourier.transform_forward()
        G_fourier_img = G_fourier.transform_forward()
        B_fourier_img = B_fourier.transform_forward()
        if self.debug:
            RGB_fourier = self.cs.combine(R_fourier_img,G_fourier_img,B_fourier_img)
            cv2.imwrite("images/fourier/"+self.filename+"_fourier.jpg",RGB_fourier)

        R_fourier_img_filtered = R_fourier.filter(mask)
        G_fourier_img_filtered = G_fourier.filter(mask)
        B_fourier_img_filtered = B_fourier.filter(mask)
        if self.debug:
            RGB_fourier_filtered = self.cs.combine(R_fourier_img_filtered,G_fourier_img_filtered,B_fourier_img_filtered)
            cv2.imwrite("images/fourier/"+self.filename+"_filtered.jpg",RGB_fourier_filtered)

        R_recovered = R_fourier.transform_backward()
        G_recovered = G_fourier.transform_backward()
        B_recovered = B_fourier.transform_backward()
        if self.debug:
            cv2.imwrite("images/fourier/split/"+self.filename+"_red_recovered.jpg",R_recovered)
            cv2.imwrite("images/fourier/split/"+self.filename+"_green_recovered.jpg",G_recovered)
            cv2.imwrite("images/fourier/split/"+self.filename+"_blue_recovered.jpg",B_recovered)
        
        recombined = ColorSplitter(self.img,self.filename).combine(R_recovered,G_recovered,B_recovered)
        if self.debug:
            cv2.imwrite("images/fourier/"+self.filename+"_recombined.jpg",recombined)
        return recombined

    
    

    
