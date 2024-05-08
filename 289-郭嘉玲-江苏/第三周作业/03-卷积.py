import cv2
import numpy as np

img = cv2.imread("lenna.png", 0)

# sobel函数求导后会有负数或者大于255的值，sobel建立的图像位数不够，会有截断
# uint8：8位无符号的数，即2^8-1=255，在[0,255]之间
# 要使用16位有符号的数据类型,2^16-1=32767,-2^16-1=-32768，在[-32768,32767]之间
# x方向进行边缘检测，y方向进行边缘检测

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

# 用convertScaleAbs()将其转回原来的uint8形式
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

# sobel对x,y两个方向进行计算，还需要用addWeighted()函数将齐组合起来
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("absX", absX)
cv2.imshow("absY", absY)
cv2.imshow("Result", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()