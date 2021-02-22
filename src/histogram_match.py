from histogram_intensity import HistogramIntensity
from matplotlib import pyplot as plt
import numpy as np
import cv2
import copy

def binary_search(arr,val):
    if len(arr) <= 2:
        return arr[0]
    m = int(len(arr)/2)
    if arr[m]==val:
        return val
    if arr[m] < val:
        return binary_search(arr[m:],val)
    elif arr[m] > val:
        return binary_search(arr[0:m],val)

class HistogramMatch:
    x_axis = np.array([i for i in range(256)])
    def __init__(self,src_filename,match_filename):
        self.src_histogram = HistogramIntensity(src_filename)
        self.match_histogram = HistogramIntensity(match_filename)
        self.outfilename = src_filename+"_xXx_"+match_filename+".jpg"
    
    def M(self,g):
        src_cdf_val = self.src_histogram.cdf_norm[g]
        dst_cdf_val = binary_search(self.match_histogram.cdf_norm,src_cdf_val)
        idx = np.where(self.match_histogram.cdf_norm == dst_cdf_val)[0][0]
        return idx
    
    def match_histograms(self):
        ##load the source image
        self.mapped_src = copy.deepcopy(self.src_histogram.img)
        ##get replace v in hsv with mapped v
        self.mapped_src[:,:,2] = np.vectorize(self.M)(self.mapped_src[:,:,2]).astype(np.float32)
        ##convert to rgb
        self.equalized_image = cv2.cvtColor(self.mapped_src, cv2.COLOR_HSV2RGB)
        cv2.imwrite("images/matched/"+self.outfilename,self.equalized_image)
        
        ## prep histogram data
        self.matched_histogram, bin_edges = np.histogram(self.mapped_src[:,:,2],bins=self.x_axis)
        hist_max = max(self.matched_histogram)
        self.matched_histogram = np.insert(self.matched_histogram,0,0)
        
        self.matched_cdf = np.cumsum(self.matched_histogram)
        cmax = max(self.matched_cdf)
        self.cdf_norm_mapped = hist_max*np.array([x/cmax for x in self.matched_cdf])
        
    
    def plot_histograms(self):
        fig,axes = plt.subplots(1,3,figsize=(10,3))
        axes[0].plot(self.x_axis,self.src_histogram.histogram)
        axes[0].plot(self.x_axis,self.src_histogram.cdf_norm*self.src_histogram.hist_max)
        axes[0].set_title(self.src_histogram.filename + " original")
        
        axes[1].plot(self.x_axis,self.match_histogram.histogram)
        axes[1].plot(self.x_axis,self.match_histogram.cdf_norm*self.match_histogram.hist_max)
        axes[1].set_title(self.match_histogram.filename + " original")

        axes[2].plot(self.x_axis,self.matched_histogram)
        axes[2].plot(self.x_axis,self.cdf_norm_mapped)
        axes[2].set_title(self.src_histogram.filename + " matched")
        plt.savefig("images/histograms/"+ self.outfilename)
        plt.close()
