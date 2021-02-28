from src.image_utils import ImageUtils
import numpy as np

class FilterBase:
    @staticmethod
    def filter(window_func,img,n,m,filename=None):
        new_img = np.ndarray(img.shape)
        if len(img.shape) == 3:
            new_img = FilterBase.filter_color(window_func,img,n,m)
        elif len(img.shape) == 2:
            new_img = FilterBase.filter_flat(window_func,img,n,m)

        if filename is not None:
            cv2.imwrite("images/filtered/"+filename+"_amf.jpg",new_img)
        return new_img

    @staticmethod
    def filter_flat(window_func,img,n,m):
        N = img.shape[0]
        M = img.shape[1]
        b_img = ImageUtils.pad_to_size(img,(N+n,M+m))
        new_img = np.ndarray(b_img.shape)
        for i in range(n//2,N+n//2):
            for j in range(m//2,M+m//2):
                new_img[i,j] = window_func(b_img,i,j,n,m)

        return new_img[n//2:N+n//2,m//2:M+m//2]
    @staticmethod
    def filter_color(window_func,img,n,m):
        new_img = np.ndarray(img.shape)
        new_img[:,:,0] = FilterBase.filter_flat(window_func,img[:,:,0],n,m)
        new_img[:,:,1] = FilterBase.filter_flat(window_func,img[:,:,1],n,m)
        new_img[:,:,2] = FilterBase.filter_flat(window_func,img[:,:,2],n,m)
        return new_img
        
    

