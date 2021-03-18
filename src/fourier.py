import numpy as np
import cv2

"""
ASSUMES ALL INPUT IMAGES ARE EITHER GRAYSCALE OR HSV
"""
class FourierTransform:
    @staticmethod
    def transform(img, filename=None):
        if len(img.shape) == 3:
            return FourierTransform.transform_color(img, filename)
        elif len(img.shape) == 2:
            return FourierTransform.transform_flat(img, filename)

    @staticmethod
    def transform_color(img, filename=None):
        new_img = np.ndarray(img.shape, dtype="complex64")
        new_img[:, :, 0] = img[:,:,0]
        new_img[:, :, 1] = img[:,:,1]
        new_img[:, :, 2] = FourierTransform.transform_flat(img[:,:,2], filename=filename)
        return new_img

    @staticmethod
    def transform_flat(img, filename=None):
        f_img = FourierTransform.reshape_fourier(img)
        f_img = np.nan_to_num(f_img, copy=True, nan=0.0, posinf=255, neginf=0) 
        f_img = np.fft.fft2(f_img)

        if filename is not None:
            vf_img = FourierTransform.get_viewable_fourier(f_img)
            cv2.imwrite("images/fourier/" + filename + ".jpg", vf_img)
        return f_img

    @staticmethod
    def inverse_transform_flat(img):
        # img = FourierTransform.reshape_fourier(img)
        return np.abs(np.fft.ifft2(img))

    @staticmethod
    def inverse_transform_color(img):
        new_img = np.ndarray(img.shape)
        Himg = img[:, :, 0]
        Simg = img[:, :, 1]
        Vimg = img[:, :, 2]
        new_img[:, :, 0] = abs(Himg)
        new_img[:, :, 1] = abs(Simg)
        new_img[:, :, 2] = FourierTransform.inverse_transform_flat(Vimg)
        return new_img

    @staticmethod
    def inverse_transform(img,filename=None):
        new_img = np.ndarray(img.shape)
        if len(img.shape) == 3:
            new_img = FourierTransform.inverse_transform_color(img)
        elif len(img.shape) == 2:
            new_img = FourierTransform.inverse_transform_flat(img)

        if filename is not None:
            cv2.imwrite("images/fourier/"+filename+"_recovered.jpg",new_img)
        return new_img

    @staticmethod
    def reshape_fourier(img):
        N = img.shape[0]
        M = img.shape[1]
        reshape_basis = np.array(
            [[(-1) ** (i + j) for j in range(M)] for i in range(N)]
        )
        if len(img.shape)==3:
            new_img = np.ndarray(img.shape,dtype=img.dtype)
            new_img[:,:,0] = img[:,:,0]
            new_img[:,:,1] = img[:,:,1]
            new_img[:,:,2] = np.multiply(img[:,:,2],reshape_basis)
            return new_img
        else:
            return np.multiply(img,reshape_basis)

    @staticmethod
    def get_viewable_fourier(img):
        fmax = np.amax(img)
        c = 255 / np.log(1 + np.abs(fmax))
        return c * np.log(1 + np.abs(img))
