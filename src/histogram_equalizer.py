import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt


class HistogramEqualizer:
    @staticmethod
    def equalize(img, filename=False):
        N = img.shape[0]
        M = img.shape[1]
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hist, cdf = HistogramEqualizer.get_hist_and_cdf(img[:, :, 2])
        cdf_max = np.max(cdf)
        cdf_min = np.min(cdf)
        cdf_norm = np.array([x / cdf_max for x in cdf])

        H = lambda v: round(255 * (cdf[v] - cdf[0]) / (cdf_max - cdf_min), 0)
        vector_H = np.vectorize(H)

        mapped_img = copy.deepcopy(img)
        mapped_img[:, :, 2] = vector_H(mapped_img[:, :, 2]).astype(np.float32)
        mapped_img = cv2.cvtColor(mapped_img, cv2.COLOR_HSV2RGB)
        
        if filename != False:
            HistogramEqualizer.plot_histograms(img, mapped_img, filename)
            cv2.imwrite("images/equalized/"+filename+".jpg",mapped_img)

        return mapped_img

    @staticmethod
    def get_hist_and_cdf(img):
        x_axis = [i for i in range(256)]
        hist, bins = np.histogram(img, bins=x_axis)
        hist = np.insert(hist, 0, 0)
        return hist, np.cumsum(hist)

    @staticmethod
    def plot_histograms(img, mapped_img, filename):
        x_axis = [i for i in range(256)]

        og_hist, og_cdf = HistogramEqualizer.get_hist_and_cdf(img[:, :, 2])
        new_hist, new_cdf = HistogramEqualizer.get_hist_and_cdf(mapped_img[:, :, 2])

        og_hist_max, new_hist_max = np.max(og_hist), np.max(new_hist)
        og_cdf_max, new_cdf_max = np.max(og_cdf), np.max(new_cdf)

        og_cdf_norm = np.array([i / og_cdf_max for i in og_cdf])
        new_cdf_norm = np.array([i / new_cdf_max for i in new_cdf])

        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        ax[0].plot(x_axis, og_hist)
        ax[0].plot(x_axis, og_cdf_norm * og_hist_max)
        ax[0].set_title(filename + "original")

        ax[1].plot(x_axis, new_hist)
        ax[1].plot(x_axis, new_cdf_norm * new_hist_max)
        ax[1].set_title(filename + " equalized")
        plt.savefig("images/histograms/" + filename + ".jpg")
        plt.close()
