from src.histogram_equalizer import HistogramEqualizer
import copy
import cv2
import numpy as np
from matplotlib import pyplot as plt


def binary_search(arr, val):
    if len(arr) <= 2:
        return arr[0]
    m = int(len(arr) / 2)
    if arr[m] == val:
        return val
    if arr[m] < val:
        return binary_search(arr[m:], val)
    elif arr[m] > val:
        return binary_search(arr[0:m], val)


class HistogramMatcher:
    @staticmethod
    def match(src_img, dst_img, filenames=None):

        src_img = np.uint8(src_img)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)

        dst_img = np.uint8(dst_img)
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2HSV)

        src_hist, src_cdf = HistogramEqualizer.get_hist_and_cdf(src_img)
        dst_hist, dst_cdf = HistogramEqualizer.get_hist_and_cdf(dst_img)

        src_cdf_max, dst_cdf_max = np.max(src_cdf), np.max(dst_cdf)

        src_cdf_norm = np.array([x / src_cdf_max for x in src_cdf])
        dst_cdf_norm = np.array([x / dst_cdf_max for x in dst_cdf])

        M = lambda g: np.where(
            dst_cdf_norm == binary_search(dst_cdf_norm, src_cdf_norm[g])
        )[0]

        mapped_img = copy.deepcopy(src_img)
        mapped_img[:, :, 2] = np.vectorize(M)(mapped_img[:, :, 2])
        mapped_img = cv2.cvtColor(mapped_img, cv2.COLOR_HSV2RGB)
        if filenames is not None:
            HistogramMatcher.plot_histograms(src_img, dst_img, mapped_img, filenames[0], filenames[1])
            cv2.imwrite("images/matched/"+filenames[0]+".jpg",mapped_img)
        return mapped_img

    @staticmethod
    def plot_histograms(src_img, dst_img, mapped_img, filename1, filename2):
        x_axis = np.array([i for i in range(256)])

        src_hist, src_cdf = HistogramEqualizer.get_hist_and_cdf(src_img[:, :, 2])
        dst_hist, dst_cdf = HistogramEqualizer.get_hist_and_cdf(dst_img[:, :, 2])
        map_hist, map_cdf = HistogramEqualizer.get_hist_and_cdf(mapped_img[:, :, 2])

        src_hist_max, dst_hist_max, map_hist_max = (
            np.max(src_hist),
            np.max(dst_hist),
            np.max(map_hist),
        )
        src_cdf_max, dst_cdf_max, map_cdf_max = (
            np.max(src_cdf),
            np.max(dst_cdf),
            np.max(map_cdf),
        )

        src_cdf_norm = np.array([i / src_cdf_max for i in src_cdf])
        dst_cdf_norm = np.array([i / dst_cdf_max for i in dst_cdf])
        map_cdf_norm = np.array([i / map_cdf_max for i in map_cdf])

        fig, ax = plt.subplots(1, 3, figsize=(10, 3))
        ax[0].plot(x_axis, src_hist)
        ax[0].plot(x_axis, src_cdf_norm * src_hist_max)
        ax[0].set_title(filename1 + " original")

        ax[1].plot(x_axis, dst_hist)
        ax[1].plot(x_axis, dst_cdf_norm * dst_hist_max)
        ax[1].set_title(filename2 + " original")

        ax[2].plot(x_axis, map_hist)
        ax[2].plot(x_axis, map_cdf_norm * map_hist_max)
        ax[2].set_title("matched")

        plt.savefig("images/histograms/" + filename1 + "_xXx_" + filename2 + ".jpg")
        plt.close()
