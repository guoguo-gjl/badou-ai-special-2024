#手写实现灰度化
from skimage.color import rgb2gray
from PIL import Image
import cv2 #导入cv2
import numpy as np #导入numpy
import matplotlib.pyplot as plt
img = cv2.imread("lenna.png") #读取图片
h,w = img.shape[:2] #读取图片高宽
img_gray = np.zeros([h,w],img.dtype) #新建一个高宽大小一致的全零数组，数据类型与原始图像一致

# 手动实现灰度化
for i in range(h):
    for j in range(w): # 遍历图像的像素点
        m = img[i,j] # m是图片上的像素，数值是rgb数组？
        img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3) #计算灰度值。注意大坑：cv是BGR，注意转换顺序
print(img_gray)
print("image show gray:%s"%img_gray) #将灰度化后的图像矩阵转为字符串
# cv2.imshow("image show gray", img_gray)

# 显示图片
plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("---image lenna---")

# 第三方库实现办法灰度化
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img

plt.subplot(222)
plt.imshow(img_gray,cmap='gray')
print("---image gray---")
print(img_gray)

rows,cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray[i,j]<= 0.5):
            img_gray[i,j]=0
        else:
            img_gray[i,j]=1

# 二值化
img_binary = np.where(img_gray>= 0.5,1,0)
print("---image binary---")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
plt.show()

