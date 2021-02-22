import cv2
import numpy as np

filename = "images/some_bish"
filetype = ".jpg"
sharpened_id = "_sharpened"
blurred_id = "_blurred"
img = cv2.imread(filename + filetype)
kernel_blur = np.array([[1,2,4,2,1],[2,4,8,4,2],[4,8,16,8,4],[2,4,8,4,2],[1,2,4,2,1]])/100
# kernel_y = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
dst_blurred = cv2.filter2D(img,cv2.CV_32F,kernel_blur)
# dst_y = cv2.filter2D(img,cv2.CV_32F,kernel_y)
# dst = cv2.sqrt(dst_x*dst_x + dst_y*dst_y)
dst = img - .5*dst_blurred
cv2.imwrite(filename+sharpened_id+filetype,2*dst)  
cv2.imwrite(filename+blurred_id+filetype,dst_blurred)