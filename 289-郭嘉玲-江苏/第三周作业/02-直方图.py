import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst = cv2.equalizeHist(gray)

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("直方图均衡化", np.hstack([gray, dst]))
cv2.waitKey(0)