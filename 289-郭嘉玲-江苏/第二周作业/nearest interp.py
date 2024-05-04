import cv2 #导入cv2库
import numpy as np #导入numpy
def function(img): #定义函数img
    height,width,channels=img.shape
    # 建立的空的image,放大到800*800；np.uint8通常用于处理像素值、图像数据等需要在0到255范围内表示的数据
    emptyImage=np.zeros((800,800,channels), np.uint8)
    sh=800/height #计算缩放的高的比例
    sw=800/width #计算缩放的宽的比例
    for i in range(800):
        for j in range(800):
            # 将采样后的图像像素坐标除以采样比例，再四舍五入，就得到了采样后像素在原图像的位置
            # x,y表示原始图像上坐标
            x=int(i/sh+0.5)  # int是向下取整，所以四舍五入要加0.5
            y=int(j/sw+0.5)
            emptyImage[i, j]=img[x, y]
    return emptyImage
# cv2.resize(img, (800,800,c), near/bin)
img=cv2.imread("lenna.png")
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)
