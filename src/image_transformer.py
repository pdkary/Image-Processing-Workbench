import cv2


class ImageTransformer:
    def __init__(self, img):
        self.img = img

    def transform(self, func, *args, **kwargs):
        self.img = func(self.img, *args, **kwargs)
        return self

    def write(self, filename):
        cv2.imwrite(filename, self.img)
