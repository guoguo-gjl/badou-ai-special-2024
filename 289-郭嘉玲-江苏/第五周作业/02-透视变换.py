# 透视变换原理
'''
取原图像和目标图像的点，计算得到坐标转换矩阵
'''

# 01-构造WarpPerspectiveMatrix函数

import numpy as np
import cv2

def WarpPerspectiveMatrix(src, dst):
    # 断言。保证dst和src的行数大于等于4
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    nums = src.shape[0]
    # 创建全0数组，np.zeros(行，列），构建了2nums*8的矩阵A，2nums*1的矩阵B
    A = np.zeros((2 * nums, 8))
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        # 选所有行的数据
        A_i = src[i, :]
        B_i = dst[i, :]
        # 根据源点和目标点坐标，填充矩阵A和向量B
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i] = B_i[0]
        A[2 * i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i+1] = B_i[1]
    A = np.mat(A)
    # 用A.I求出A的逆矩阵，之后与B相乘，求出warpMatrix
    warpMatrix = A.I * B
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

if __name__ == '__main__':
    print('warpMatrix')
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)
    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)
    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)

# 02-调包cv2.getPerspectiveTransform()

img = cv2.imread('photo1.jpg')
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
# 得到透视变换矩阵
m = cv2.getPerspectiveTransform(src, dst)
result4 = cv2.warpPerspective(img, m, (337, 448))
cv2.imshow('src', img)
cv2.imshow('result', result4)
cv2.waitKey(0)

# 03-寻找顶点
img = cv2.imread('photo1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 高斯模糊
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# 图像膨胀操作
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT), (3, 3))
# canny边缘检测，用来检测膨胀后图像的边缘
edged = cv2.Canny(dilate, 30, 120, 3)
# 查找膨胀后图像的轮廓
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0]
docCnt = None

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            docCnt = approx
            break
for peak in docCnt:
    peak = peak[0]
    cv2.circle(img, tuple(peak), 10, (255, 0, 0))
cv2.imshow('img', img)
cv2.waitKey(0)





