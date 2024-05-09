import numpy as np
import cv2
from numpy import shape
import random
from skimage import util

def fun1(src, percetage):
    noiseimg = src
    noisenum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(noisenum):
        randx = random.randint(0, src.shape[0] -1)
        randy = random.randint(0, src.shape[1] -1)
        # random.random()生成随机浮点数
        if random.random()< 0.5: # 掷色子，判断是赋椒噪声还是盐噪声
            noiseimg[randx,randy] = 0
        else:
            noiseimg[randx, randy] = 255
    return noiseimg

img = cv2.imread("lenna.png", 0)
img1 = fun1(img, 0.2)

img = cv2.imread("lenna.png")
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("原图", img2)
cv2.imshow("lenna椒盐噪声", img1)
cv2.waitKey(0)

# 泊松噪声
img = cv2.imread("lenna.png", 0)
poisson_img = util.random_noise(img, mode='poisson')
cv2.imshow("lenna_poisson",poisson_img)
cv2.waitKey(0)

# 高斯噪声
img = cv2.imread("lenna.png", 0)
gaussian_noise_img = util.random_noise(img, mode="gaussian")
cv2.imshow("lenna_gaussian", gaussian_noise_img)
cv2.waitKey(0)

# 椒盐噪声
img = cv2.imread("lenna.png", 0)
sp_noise_img = util.random_noise(img, mode="s&p")
cv2.imshow("lenna_sp", sp_noise_img)
cv2.waitKey(0)

# 椒燥声
img = cv2.imread("lenna.png", 0)
pepper_noise_img = util.random_noise(img, mode="pepper")
cv2.imshow("lenna_pepper", pepper_noise_img)
cv2.waitKey(0)

# 盐燥声
img = cv2.imread("lenna.png", 0)
salt_noise_img = util.random_noise(img, mode="salt")
cv2.imshow("lenna_salt", salt_noise_img)
cv2.waitKey(0)

# 均匀噪声
img = cv2.imread("lenna.png", 0)
speckle_noise_img = util.random_noise(img, mode="speckle")
cv2.imshow("lenna_speckle", speckle_noise_img)
cv2.waitKey(0)

