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
        f = lambda x: 255 if x >= p else 0
        return IntensityTransformer.transform(img,f,filename+"_step")


