'''
1.k-means算法原理：
将数据集划分为k个簇，使得每个数据点都属于最近的簇，且簇的中心是所有数据点的平均值
算法是基于迭代优化的，每个优化步骤会更新簇的中心点，直到达到收敛条件
2.k-means算法计算步骤：
    01初始化：选择要将数据集分成k个簇，随机选择k个数据点作为初始簇中心
    02分配：将每个数据点分配到距离其最近的簇中心，每个数据点只属于一个簇
    03更新：根据分配的数据点更新簇中心点，通过计算每个簇的数据点平均值来实现
    04重复：重复02和03步骤，直到簇中心点不再发生变化，或达到预定的迭代次数
    05输出：得到k个簇和每个簇的中心点
3.k-means算法优缺点
    优点：易于实现和理解，计算效率高，适用于大规模数据集，适用于高维数据
    缺点：需要手动指定簇的个数k，否则会影响最终聚类的效果
         对于非凸的簇结构，k-means表现效果不佳，容易陷入局部最优解
         初始簇中心点的随机选择可能导致不同的聚类结果
4.k-means聚类在图像领域的应用
    可通过k-means聚类算法实现图像分割、图像聚类、图像识别等操作
    通过k-means将像素点聚成k个簇，使用簇内的质心点替换簇内所有像素点，能够在不改变分辨率的情况下量化压缩图像颜色，实现颜色层级划分
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 01-cv2.kmeans运用分割lenna

img = cv2.imread('lenna.png', 0)
print(img.shape)
# 获取高宽
rows, cols = img.shape[:]
# 将二维数组转为一维数组，形状为(rows*cols,1)
data = img.reshape((rows * cols, 1))
data = np.float32(data)
# 设置停止准则
'''
停止准则几种模式
    cv2.TERM_CRITERIA_EPS 精确度满足epsilon停止
    cv2.TERM_CRITERIA_MAX_ITER 迭代次数超过max_iter停止
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER两者都用，任意满足一个就结束
'''
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# flags表示初始中心的选择，两种方法：
# 指定的选择方法：cv2.KMEANS_PP_CENTERS;
# 随机的选择方法：cv2.KMEANS_RANDOM_CENTERS
flags = cv2.KMEANS_RANDOM_CENTERS

# k-means聚类，聚成4类。将lenna图片的多个灰度级压缩成4个灰度级
'''
data:输入的数据；
4：聚类中心数量；
None：初始化聚类中心的初始中心点，None表示算法自动选择初始中心点
criterian：迭代停止的准则
10：表示重复运行次数
flags：表示额外的标志
compactness, labels, centers三值分别表示每个点到聚类中心距离的平方和、每个数据点所属的聚类标签、聚类中心点的坐标值
'''

# 10是重复kmeans算法次数，返回结果最好的一次结果
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

# 一维度转回二维度
dst = labels.reshape((img.shape[0], img.shape[1]))

# 设置中文显示的字体为'SimHei'
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
    # 创建一个包含 1 行 2 列子图的图像，并选择第 i+1 个子图
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray'),
    # 为当前子图添加标题，标题内容为titles[i]
    plt.title(titles[i])
    # 隐藏子图的坐标轴标记
    plt.xticks([]), plt.yticks([])
plt.show()

# 02-运动员信息聚类

from sklearn.cluster import KMeans

X = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
    ]

# 载入数据集X，并且将聚类的结果赋值给y_pred
clf = KMeans(n_clusters=3)
y_pred = clf.fit_transform(X)
print(clf)
print('y_pred = ', y_pred)

# 可视化绘图
x = [n[0] for n in X]
print(x)
y = [n[1] for n in X]
print(y)
plt.scatter(x, y, c=y_pred, marker='x')
plt.title('Kmeans-Basketball Data')
plt.xlabel('assists_per_minute')
plt.ylabel('points_per_minute')
plt.legend(['A', 'B', 'C'])
plt.show()

# 03-K-means-RGB
img = cv2.imread('lenna.png')
data = img.reshape((-1, 3))
data = np.float32(data)
criteria3 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
# 聚成2类
compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)
# 聚成4类
compactness, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)
# 聚成8类
compactness, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)
# 聚成16类
compactness, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)
# 聚成64类
compactness, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

# 图像转回uint8二维类型
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape((img.shape))

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape((img.shape))

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape((img.shape))

centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape((img.shape))

centers64 = np.uint8(centers64)
res = centers64[labels64.flatten()]
dst64 = res.reshape((img.shape))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

plt.rcParams['font.sans-serif'] = ['SimHei']
titles = [u'原始图像', u'聚类图像k=2', u'聚类图像 K=4', u'聚类图像 K=8', u'聚类图像 K=16',  u'聚类图像 K=64']
images = [img, dst2, dst4, dst8, dst16, dst64]
for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()