import cv2


class ImageTransformer:
    def __init__(self, img,debug=False):
        self.img = img
        self.debug = debug

    def transform(self, func, *args, **kwargs):
        if self.debug:
            print("beginning {}".format(func.__name__))
        self.img = func(self.img, *args, **kwargs)
        return self

    def get(self):
        return self.img
        
    def write(self, filename):
        cv2.imwrite(filename, self.img)
