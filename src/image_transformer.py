import cv2


class ImageTransformer:
    def __init__(self, img,debug=False):
        self.img = img
        self.debug = debug

    def transform(self, func, *args, **kwargs):
        if self.debug:
            print("beginning {}".format(func.__name__))
            print("| shape before: "+str(self.img.shape))
        self.img = func(self.img, *args, **kwargs)
        if self.debug:
            print("| shape after: "+str(self.img.shape))
        return self

    def get(self):
        return self.img
        
    def write(self, filename):
        cv2.imwrite(filename, self.img)
