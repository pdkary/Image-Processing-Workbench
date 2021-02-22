import threading
from tqdm import tqdm
import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy 

class HistogramIntensity:
    x_axis = np.array([i for i in range(256)])
    def __init__(self, filename):
        self.filename = filename
        self.img = cv2.cvtColor(cv2.imread("images/"+filename+".jpg"),cv2.COLOR_RGB2BGR)
        self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2HSV)
        self.N = self.img.shape[0]
        self.M = self.img.shape[1]
        
        self.histogram, self.bin_edges = np.histogram(self.img[:,:,2],bins=self.x_axis)
        self.histogram = np.insert(self.histogram,0,0)
        self.hist_max = max(self.histogram)
        
        self.cdf = np.cumsum(self.histogram)
        self.cdf_min = min(self.cdf)
        self.cdf_max = max(self.cdf)
        self.cdf_norm = np.array([x/self.cdf_max for x in self.cdf])

    def plot_histogram(self):
        fig, ax = plt.subplots(1,2,figsize=(7,3))
        ax[0].plot(self.x_axis,self.histogram)
        ax[0].plot(self.x_axis,self.cdf_norm*self.hist_max)
        ax[0].set_title(self.filename + " original")
        
        ax[1].plot(self.x_axis,self.mapped_image_histogram)
        ax[1].plot(self.x_axis,self.cdf_norm_mapped*self.mapped_hist_max)
        ax[1].set_title(self.filename + " equalized")
        plt.savefig("images/histograms/"+self.filename+".jpg")
        plt.close()

    def H(self, v):
        return round(255*(self.cdf[v]-self.cdf[0])/(self.cdf_max - self.cdf_min),0)

    def equalize_image(self):
        vector_H = np.vectorize(self.H)
        self.mapped_image = copy.deepcopy(self.img)
        self.mapped_image[:,:,2] = vector_H(self.mapped_image[:,:,2]).astype(np.float32)
        
        self.mapped_image_histogram, bin_edges = np.histogram(self.mapped_image[:,:,2],bins=self.x_axis)
        self.mapped_image_histogram = np.insert(self.mapped_image_histogram,0,0)
        self.mapped_hist_max = max(self.mapped_image_histogram)
        
        self.cdf_mapped = np.cumsum(self.mapped_image_histogram)
        self.cdf_min_mapped = self.cdf_mapped[2]
        self.cdf_max_mapped = self.cdf_mapped[-1]
        self.cdf_norm_mapped = np.array([x/self.cdf_max_mapped for x in self.cdf_mapped])
        
        self.equalized_image = cv2.cvtColor(self.mapped_image, cv2.COLOR_HSV2RGB)
        cv2.imwrite("images/equalized/"+self.filename+".jpg",self.equalized_image)