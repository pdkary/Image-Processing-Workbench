import numpy as np
import cv2
import threading
from tqdm import tqdm


class Filter2D:
    IMAGE_PATH = "images/"
    FILTER_PATH = "images/filtered/"
    filtered = 1
    equalized = 2
    histograms = 3
    def __init__(self, filename):
        filepath = filename.split("/")
        if len(filepath)>1:
            if filepath[0] == "filtered":
                self.type = Filter2D.filtered
            if filepath[0] == "equalized":
                self.type = Filter2D.equalized
            if filepath[0] == "histogram":
                self.type = Filter2D.histograms
        else:
            self.type = 0
        self.filename = filepath[-1]
        self.img = cv2.imread(Filter2D.IMAGE_PATH+filename, 1)

    def filter(self, kernel, threads=1):
        self.kernel = kernel
        self.border_size = int((kernel.shape[0]-1)/2)
        self.bordered_img = cv2.copyMakeBorder(
            self.img, self.border_size, self.border_size, self.border_size, self.border_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        self.filtered_img = np.ndarray(self.bordered_img.shape)
        self.y_size = self.img.shape[0]
        self.x_size = self.img.shape[1]
        self.filter_threaded(threads)
        self.write()

    def write(self):
        unbordered = self.filtered_img[self.border_size:-self.border_size-1,self.border_size:-self.border_size-1]
        path = Filter2D.FILTER_PATH+self.filename
        if self.type == Filter2D.equalized:
            path +="_equalized"
        if self.type == Filter2D.filtered:
            path +="_filtered"
        if self.type == Filter2D.histograms:
            path +="_histogram"
        cv2.imwrite(path, unbordered)

    def evaluate_kernel_at(self, center_x, center_y):
        start = (center_x-self.border_size, center_y-self.border_size)
        end = (center_x+self.border_size, center_y+self.border_size)

        window = self.img[start[0]:end[0], start[1]:end[1]]
        new_pixel = [0, 0, 0]
        for i, col in enumerate(window):
            for j, val in enumerate(col):
                new_val = val*self.kernel[i][j]
                new_pixel[0] += new_val[0]
                new_pixel[1] += new_val[1]
                new_pixel[2] += new_val[2]
        return new_pixel

    def filter_indicies(self, i_list):
        j_list = [x for x in range(self.border_size, self.x_size)]
        for i in tqdm(i_list):
            for j in j_list:
                self.filtered_img[i, j] = self.evaluate_kernel_at(i, j)

    def filter_threaded(self, num_threads):
        threads = []
        for i in range(num_threads):
            i_list = [x for x in range(
                self.border_size+i, self.y_size, num_threads)]
            x = threading.Thread(target=self.filter_indicies, args=[i_list])
            threads.append(x)
            x.start()
        for thread in threads:
            thread.join()
