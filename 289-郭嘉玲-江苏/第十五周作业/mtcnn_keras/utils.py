import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from operator import itemgetter

# 计算缩放比例函数
def calculateScales(img):
    copy_img = img.copy()
    pr_scale = 1.0
    h, w, _ = copy_img.shape()
    if min(w, h) > 500:
        pr_scale = 500.0/min(h, w)
        w = int(w*pr_scale)
        h = int(h*pr_scale)
    elif max(w, h) < 500:
        pr_scale = 500.0/max(h, w)
        w = int(w*pr_scale)
        h = int(h*pr_scale)

    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h, w)
    while minl >= 12:
        scales.append(pr_scale*pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales

# 处理pnet处理后的结果
def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):

    # 交换轴
    cls_prob = np.swapaxes(cls_prob, 0, 1)
    roi = np.swapaxes(roi, 0, 2)

    # 计算步幅
    stride = 0
    if out_side != 1:
        stride = float(2*out_side - 1)/(out_side - 1)

    # 找到符合阈值的点
    (x, y) = np.where(cls_prob >= threshold)

    # 计算边界框
    boundingbox = np.array([x, y]).T
    bb1 = np.fix((stride*(boundingbox)+0)*scale)
    bb2 = np.fix((stride*(boundingbox)+11)*scale)
    boundingbox = np.concatenate((bb1, bb2), axis=1)

    # 计算偏移量
    dx1 = roi[0][x, y]
    dx2 = roi[1][x, y]
    dx3 = roi[2][x, y]
    dx4 = roi[3][x, y]
    score = np.array([cls_prob[x, y]]).T
    offset = np.array([dx1, dx2, dx3, dx4]).T

    # 调整边界框的位置
    boundingbox = boundingbox + offset*12.0*scale

    # 合并边界框和分数
    rectangles = np.concatenate((boundingbox, score), axis=1)
    # 将边界框调整为正方形
    rectangles = rect2square(rectangles)
    # 剪裁边界框
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, y2, sc])

    # 非极大值抑制， 去除重复的边界框，只保留最优的边界框
    return NMS(pick, 0.3)

# 将长方形调整为正方形
def rect2square(rectangles):
    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]
    l = np.maximum(w, h).T
    rectangles[:, 0] = rectangles[:, 0] + w*0.5 - l*0.5
    rectangles[:, 1] = rectangles[:, 1] + h*0.5 - l*0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([1], 2, axis=0).T
    return rectangles

# 非极大值抑制
def NMS(rectangles, threshold):

    # 检查空输入
    if len(rectangles) == 0:
        return rectangles

    # 初始化变量
    boxes = np.array(rectangles)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)

    # 排序边界框
    I = np.array(s.argsort())

    # 执行NMS
    pick = []
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        o = inter/(area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o<=threshold)[0]]

    # 返回结果
    result_rectangle = boxes[pick].tolist()
    return result_rectangle

