from filter2d import Filter2D
from histogram_intensity import HistogramIntensity
from histogram_match import HistogramMatch
from fourier import FourierTransform,MaskMaker
from color_fourier import ColorFourier
import numpy as np
import cv2

kernel_gauss_5x5 = np.array([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
                             [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                             [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
                             [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                             [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])

kernel_5x5_laplace = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [
                              1, 1, -24, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

kernel_3x3_mean = np.array([[1, 1, 1], [1, 8, 1], [1, 1, 1]])/16

if __name__ == '__main__':

    filename1 = "bigfoot"
    filename2 = "renaissance"

    hm = HistogramMatch(filename1,filename2)
    hm.match_histograms()
    hm.plot_histograms()

    hm.src_histogram.equalize_image()
    hm.src_histogram.plot_histogram()
    hm.match_histogram.equalize_image()
    hm.match_histogram.plot_histogram()

