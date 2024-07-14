import cv2 as cv
import numpy as np
from PIL import Image
import os.path as path
from PIL import ImageEnhance

def rotate(image):
    def rotate_bound(image, angle):
        (h, w) = image.shape[:2]
        cX, cY = h//2, w//2
        M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h*sin) + (w*cos))
        nH = int((h*cos) + (w*sin))
        M[0, 2] += (nW/2) - cX
        M[]