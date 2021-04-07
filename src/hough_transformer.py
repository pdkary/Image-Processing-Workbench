import numpy as np
import cv2
from src.edge_detector import EdgeDetector
from src.image_utils import ImageUtils
from src.intensity_transformer import IntensityTransformer
from math import pi
class HoughTransformer:
    H = {}

    @staticmethod
    def transform(img,threshold=100,filename=None):
        if len(img.shape)==3:
            return HoughTransformer.transform_color(img,threshold,filename)
        if len(img.shap3)==2:
            return HoughTransformer.transform_flat(img,threshold,filename)
    
    def transform_color(img, threshold=100, filename=None):
        grey_img = ImageUtils.convert_to_grayscale(img)
        return HoughTransformer.transform_flat(grey_img,threshold,filename)
    
    @staticmethod
    def transform_flat(img, threshold=100, filename=None):
        ## apply sobel edge detection
        sobel_x,sobel_y,mag = EdgeDetector.sobel_with_blur(img,return_edges=True)
        mag = IntensityTransformer.upper_threshold(mag,k=100)

        if filename is not None:
            cv2.imwrite("images/hough/"+filename + "_mag.jpg",mag)
        
        N = mag.shape[0]
        M = mag.shape[1]
        HoughTransformer.reset_H(N,M)
        for i in range(N):
            for j in range(M):
                if(mag[i][j] > 0):
                    for t in range(180):
                        r = int(np.round(i*np.cos(t*pi/180)+j*np.sin(t*pi/180)))
                        HoughTransformer.H[r,t]+=1
        
        ##now we have an array of (root(N*N,M*M),180) with buckets of each line usage
        line_img = np.ndarray(shape=mag.shape)
        r_range = int(np.round(np.sqrt(N*N+M*M)))
        for r in range(r_range):
            for t in range(180):
                if(HoughTransformer.H[r,t]>threshold):
                    a = np.cos(t*pi/180)
                    b = np.sin(t*pi/180)
                    x0 = a*r
                    y0 = b*r
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    cv2.line(line_img,(y1,x1),(y2,x2),255,1)
        return line_img


    @staticmethod
    def reset_H(N,M):
        r_range = int(np.round(np.sqrt(N*N+M*M)))
        HoughTransformer.H = np.zeros(shape=(r_range,180))

