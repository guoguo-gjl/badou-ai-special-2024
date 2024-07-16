import cv2
import numpy as np

# 01-关键点检测 SIFT_create()、detectAndCompute()
img = cv2.imread('mountain1.jpg')
sift = cv2.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(img, None)
img_kp = cv2.drawKeypoints(img, keypoints, img)
cv2.imshow('sift_keypoints', img_kp)
img = cv2.drawKeypoints(image=img, keypoints=keypoints, outImage=img,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))
cv2.imshow('sift_img', img)
cv2.waitKey()
cv2.destroyAllWindows()

# 02-特征匹配
def drawMatchKnn_cv2(img1_gray, img2_gray):

    # 创建新图像用于匹配绘制结果
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1+w2] = img2_gray

    # sift特征检测
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # 定义暴力匹配器进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    # 筛选匹配结果
    goodMatch = []
    for m, n in matches:
        if m.distance < 0.50*n.distance:
            goodMatch.append(m)

    # 获取匹配点索引
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    # 显示匹配结果
    cv2.namedWindow('match', cv2.WINDOW_NORMAL)
    cv2.imshow('match', vis)

img1_gray = cv2.imread('mountain1.jpg')
img2_gray = cv2.imread('mountain2.jpg')
drawMatchKnn_cv2(img1_gray, img2_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()