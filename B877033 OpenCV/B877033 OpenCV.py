import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
GREEN = [0,255,0]
img1 = cv.imread('Lenna.png')

def Tutorial():
    replicate = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REPLICATE)
    reflect = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT)
    reflect101 = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT_101)
    wrap = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_WRAP)
    constant= cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_CONSTANT,value=GREEN)
    plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
    plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
    plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
    plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
    plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
    plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
    plt.show()

def Black():
    kernel = np.ones((5,5),np.uint8)
    erosion = cv.erode(img1,kernel,iterations = 1)
    dilation = cv.dilate(img1,kernel,iterations = 1)
    opening = cv.morphologyEx(img1, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(img1, cv.MORPH_CLOSE, kernel)
    gradient = cv.morphologyEx(img1, cv.MORPH_GRADIENT, kernel)
    tophat = cv.morphologyEx(img1, cv.MORPH_TOPHAT, kernel)


    plt.subplot(231),plt.imshow(erosion),plt.title('Erosion')
    plt.xticks([]), plt.yticks([])
    plt.subplot(232),plt.imshow(dilation),plt.title('Dilation')
    plt.xticks([]), plt.yticks([])
    plt.subplot(233),plt.imshow(opening),plt.title('Opening')
    plt.xticks([]), plt.yticks([])
    plt.subplot(234),plt.imshow(closing),plt.title('Closing')
    plt.xticks([]), plt.yticks([])
    plt.subplot(235),plt.imshow(gradient),plt.title('Gradient')
    plt.xticks([]), plt.yticks([])
    plt.subplot(236),plt.imshow(tophat),plt.title('Top Hat')
    plt.xticks([]), plt.yticks([])
    plt.show()

def custom():
    kernel = np.ones((5,5),np.float32)/25
    dst1=cv.filter2D(img1,-1,kernel)
    constant1= cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_CONSTANT,value=GREEN)
    dst = cv.filter2D(constant1,-1,kernel)
    plt.subplot(221),plt.imshow(img1),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(constant1,'gray'),plt.title('CONSTANT')
    plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(dst1),plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(dst),plt.title('CONSTANT+Averaging')
    plt.xticks([]), plt.yticks([])
    plt.show()

if __name__=="__main__":
    Tutorial()
    Black()
    custom()