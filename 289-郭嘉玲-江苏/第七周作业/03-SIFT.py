import cv2
import numpy as np

# 01-SIFT关键点
img1 = cv2.imread('mountain1.jpg')
img2 = cv2.imread('mountain2.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建一个sift对象，用于检测关键点、计算描述子
sift = cv2.SIFT_create()
# sift对象检测灰度图像关键点；keypoints存储检测到的关键点，descrptor存储计算得到的描述子
keypoints1, descriptor1 = sift.detectAndCompute(img1, None)
keypoints2, descriptor2 = sift.detectAndCompute(img2, None)
# 原始彩色图像绘制关键点
img1_kp = cv2.drawKeypoints(image=img1, outImage=img1, keypoints=keypoints1,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))
img2_kp = cv2.drawKeypoints(image=img2, outImage=img2, keypoints=keypoints2,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))
# 灰度图像绘制关键点
img1 = cv2.drawKeypoints(img1_kp, keypoints1, img1_kp)
img2 = cv2.drawKeypoints(img2_kp, keypoints2, img2_kp)
cv2.imshow('sift_keypoints', img1)
cv2.imshow('sift_keypoints_color', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 02-SIFT特征匹配
def drawMatchchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1: w1+w2] = img2_gray
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]
    # 提取匹配点坐标并将第二张图片的点平移至正确位置
    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.namedWindow('match', cv2.WINDOW_NORMAL)
    cv2.imshow('match', vis)

# 读取图片
img1_gray = cv2.imread('mountain1.jpg')
img2_gray = cv2.imread('mountain2.jpg')
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)
goodMatch = []
for m, n in matches:
    if m.distance < 0.50*n.distance:
        goodMatch.append(m)
drawMatchchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])
cv2.waitKey(0)
cv2.destroyAllWindows()