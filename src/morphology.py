import numpy as np


class Morphology:

    @staticmethod
    def translation(A,dx,dy):
        ##A := list of indicies in a Structural Element
        Az = [[p[0]+dx,p[1]+dy] for p in A]
        return np.array(Az)
    
    @staticmethod
    def reflection(A,origin):
        Ar = []
        for p in A:
            dx = origin[0] - p[0]
            dy = origin[1] - p[1]
            Ar.append([origin[0]-dx,origin[1]-dy])
        return np.array(Ar)

    @staticmethod
    def erode(img,kernel):
        new_img = np.ndarray(shape=img.shape)
        N = img.shape[0]
        M = img.shape[1]
        for i in range(N):
            for j in range(M):
                offset_kernel = [(k[0]+i if k[0]+i < N else N-1,k[1]+j if k[1]+j < M else M-1) for k in kernel]
                Y = np.transpose(offset_kernel)[0]
                X = np.transpose(offset_kernel)[1]
                vals = img[Y,X]
                if np.all(vals==255):
                    new_img[i,j]=255
                else:
                    new_img[i,j]=0
        return new_img
    
    @staticmethod
    def dilate(img,kernel):
        new_img = np.ndarray(shape=img.shape)
        N = img.shape[0]
        M = img.shape[1]
        for i in range(N):
            for j in range(M):
                ##(k[n]+i,k[n]+j) for all points in kernel ( such that (0,0) becomes (i,j))
                offset_kernel = [(k[0]+i if k[0]+i < N else N-1,k[1]+j if k[1]+j < M else M-1) for k in kernel]
                Y = np.transpose(offset_kernel)[0]
                X = np.transpose(offset_kernel)[1]
                vals = img[Y,X]
                if np.any(vals==255):
                    new_img[i,j]=255
                else:
                    new_img[i,j]=0
        return new_img
    
    @staticmethod
    def open(img,kernel):
        e_img = Morphology.erode(img,kernel)
        return Morphology.dilate(img,kernel)

    @staticmethod
    def open_n_times(img,kernel,n):
        curr_img = img
        for i in range(n):
            curr_img = Morphology.open(curr_img,kernel)
        return curr_img

    @staticmethod
    def close_n_times(img,kernel,n):
        curr_img = img
        for i in range(n):
            curr_img = Morphology.close(curr_img,kernel)
        return curr_img
    
    @staticmethod
    def close(img,kernel):
        d_img = Morphology.dilate(img,kernel)
        return Morphology.erode(img,kernel)




