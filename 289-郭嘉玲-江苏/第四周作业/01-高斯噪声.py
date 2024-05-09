import numpy as np
import cv2
import random
from numpy import shape

def GaussianNoise(src,means,sigma,percetage):
    NoiseImg = src
    NoiseNum = int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        # 求要加噪的像素坐标，图片边缘不处理，故-1
        randx = random.randint(0,src.shape[0]-1)
        randy = random.randint(0,src.shape[1]-1)
        # 给要加噪的像素坐标加噪
        NoiseImg[randx, randy] = NoiseImg[randx,randy] + random.gauss(means,sigma)
        # 缩放加噪值在[0,255]之间
        if NoiseImg[randx, randy] < 0:
            NoiseImg[randx, randy] = 0
        elif NoiseImg[randx, randy] > 255:
            NoiseImg[randx, randy] = 255
    return NoiseImg
img = cv2.imread("lenna.png", 0)
img1 = GaussianNoise(img, 2, 10, 1)
img = cv2.imread("lenna.png")
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("原图", img2)
cv2.imshow("lenna高斯", img1)
cv2.waitKey(0)
