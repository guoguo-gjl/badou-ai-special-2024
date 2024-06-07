import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# 01-调接口
pic = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
cv2.imshow("canny", cv2.Canny(gray, 200, 300))
cv2.waitKey()
cv2.destroyAllWindows()

# 02-优化程序

def CannyThreshold(lowThreshold):
    detected_edges = cv2.Canny(gray,
                               lowThreshold,
                               lowThreshold * ratio,
                               apertureSize=kernel_size)
    # '按位与'操作
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    cv2.imshow('canny result', dst)
if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    # 调节杆的最大值和最小值
    lowThreshold = 0
    maxThreshold = 100
    ratio = 3
    kernel_size = 3
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('canny result')
    # createTrackbar()
    # 第一个参数是调节杠最小值
    # 第二个窗口名
    # 第三个最小值
    # 第四个最大值
    # 第五个CannyThreshold是回调函数，这个回调函数会根据滑动条的值来调整图像处理的参数，以实现动态调节算法参数的目的。
    cv2.createTrackbar('Min threshold', 'canny result', lowThreshold, maxThreshold, CannyThreshold)
    CannyThreshold(0)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()


# 03-几种边缘检测的算子（soble-laplace-canny）
img = cv2.imread('lenna.png', 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sobel 算子
img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
img_sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=3)
# laplace 算子
img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)
# canny 算子
img_canny = cv2.Canny(img_gray, 100, 150)
# 展示
plt.subplot(231), plt.imshow(img_gray, 'gray'), plt.title('original')
plt.subplot(232), plt.imshow(img_sobel_x, 'gray'), plt.title('sobel_x')
plt.subplot(233), plt.imshow(img_sobel_y, 'gray'), plt.title('sobel_y')
plt.subplot(236), plt.imshow(img_sobel, 'gray'), plt.title('sobel')
plt.subplot(234), plt.imshow(img_laplace, 'gray'), plt.title('laplace')
plt.subplot(235), plt.imshow(img_canny, 'gray'), plt.title('canny')
plt.show()









