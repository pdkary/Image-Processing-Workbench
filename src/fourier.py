import numpy as np
import cv2


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
        Rimg = img[:, :, 0]
        Gimg = img[:, :, 1]
        Bimg = img[:, :, 2]
        if filename != None:
            rf = "split/" + filename + "_red"
            gf = "split/" + filename + "_green"
            bf = "split/" + filename + "_blue"
        else:
            rf, bf, gf = None, None, None
        new_img[:, :, 0] = FourierTransform.transform_flat(Rimg, rf)
        new_img[:, :, 1] = FourierTransform.transform_flat(Gimg, gf)
        new_img[:, :, 2] = FourierTransform.transform_flat(Bimg, bf)
        return new_img

    @staticmethod
    def transform_flat(img, filename=None):
        N = img.shape[0]
        M = img.shape[1]
        reshape_basis = np.array(
            [[(-1) ** (i + j) for j in range(M)] for i in range(N)]
        )
        img = np.multiply(img, reshape_basis)

        f_img = np.fft.fft2(img)

        if filename is not None:
            vf_img = FourierTransform.get_viewable_fourier(f_img)
            cv2.imwrite("images/fourier/" + filename + ".jpg", vf_img)
        return f_img

    @staticmethod
    def inverse_transform_flat(img):
        return np.abs(np.fft.ifft2(img))

    @staticmethod
    def inverse_transform_color(img):
        new_img = np.ndarray(img.shape)
        Rimg = img[:, :, 0]
        Gimg = img[:, :, 1]
        Bimg = img[:, :, 2]
        new_img[:, :, 0] = FourierTransform.inverse_transform_flat(Rimg)
        new_img[:, :, 1] = FourierTransform.inverse_transform_flat(Gimg)
        new_img[:, :, 2] = FourierTransform.inverse_transform_flat(Bimg)
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
    def get_viewable_fourier(img):
        fmax = np.amax(img)
        c = 255 / np.log(1 + np.abs(fmax))
        return c * np.log(1 + np.abs(img))
