import numpy as np
import cv2

class IntensityTransformer:

    @staticmethod
    def transform(img,func,filename=None):
        if len(img.shape)==3:
            return IntensityTransformer.transform_color(img,func,filename)
        if len(img.shape)==2:
            return IntensityTransformer.transform_flat(img,func,filename)
    
    @staticmethod
    def transform_color(img,func,filename=None):
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:,:,2] = np.vectorize(func)(img[:,:,2])
        img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
        if filename is not None:
            cv2.imwrite("images/filtered/"+filename+"_mapped.jpg",img)
        return img

    @staticmethod
    def transform_flat(img,func,filename=None):
        img = np.vectorize(func)(img)
        if filename is not None:
            cv2.imwrite("images/filtered/"+filename+"_mapped.jpg",img)
        return img
    
    @staticmethod
    def quantize(img,q,filename=None):
        inc = 255//q
        levels = [inc*i for i in range(q)]

        def get_q(x):
            if x==0:
                return 0
            for i in range(q):
                if levels[i] > x:
                    if i==1 or i==0:
                        return 0
                    else:
                        return levels[i-1]
            return levels[-1]
        
        return IntensityTransformer.transform(img,get_q,filename)
    
    @staticmethod
    def binary_step(img,p,filename=None):
        newfilename = filename+"_step" if filename is not None else None
        f = lambda x: 255 if x >= p else 0
        return IntensityTransformer.transform(img,f,newfilename)

    @staticmethod
    def map_intensities_to_viewable(img):
        max_val = np.max(img)
        min_val = np.min(img)
        f = lambda x: 255*(x-min_val)/(max_val-min_val)
        return np.vectorize(f)(img)
    
    @staticmethod
    def sigmoid(img,filename=None):
        newfilename = filename+"_sigmoid" if filename is not None else None
        f = lambda x: round(263/(1+np.exp(-(x-128)/30)))-4
        return IntensityTransformer.transform(img,f,newfilename)
    
    @staticmethod
    def shift(img,k=1,filename=None):
        newfilename = filename+"_shift_by_"+str(k) if filename is not None else None
        def f(x):
            out = x+k
            if out < 0:
                out=0
            if out > 255:
                out=255
            return out
        return IntensityTransformer.transform(img,f,newfilename)
    
    @staticmethod
    def linear_cutoff(img,k=250,filename=None):
        newfilename = filename+"_cutoff_at_"+str(k) if filename is not None else None
        def f(x):
            if x >= k:
                return k
            return x
        return IntensityTransformer.transform(img,f,newfilename)

    @staticmethod
    def upper_threshold(img,k=220,filename=None):    
        newfilename = filename+"_upper_t_"+str(k) if filename is not None else None

        f = lambda x: x if x >= k else 0
        return IntensityTransformer.transform(img,f,newfilename)
    
    @staticmethod
    def lower_threshold(img,k=220,filename=None):    
        newfilename = filename+"_lower_t_"+str(k) if filename is not None else None

        f = lambda x: x if x <= k else 0
        return IntensityTransformer.transform(img,f,newfilename)
    
    @staticmethod
    def band_threshold(img,k1=100,k2=200,filename=None):
        newfilename = filename+"_band_t_"+str(k1)+"_"+str(k2) if filename is not None else None
        f = lambda x: x if (x>=k1 and x<=k2) else 0
        return IntensityTransformer.transform(img,f,newfilename)