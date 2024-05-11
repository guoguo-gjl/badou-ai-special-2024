# 01-PCA_sklearn

import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1, 2, 66, -1], [-2, 44, 78, 0], [-2, 9, 88, 80], [3, 5, 66, 3]])
pca = PCA(n_components=2)
pca.fit(X)
newX = pca.fit_transform(X)
print(newX)

# 02-PCA_numpy

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self,X):
        self.n_features = X.shape[1]
        # 求协方差矩阵
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T, X) / X.shape[0]  #np.dot是什么？
        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        # 对X降维
        return np.dot(X, self.components_)

pca = PCA(n_components=2)
X = np.array(
    [[-1, 2, 66, -1],
     [-2, 6, 58, -1],
     [-3, 8, 45, -2],
     [1, 9, 36, 1],
     [2, 10, 62, 1],
     [3, 5, 83, 2]]
)
newX = pca.fit_transform(X)
print(newX)

# 03-numpy-detail
class CPCA(object):
    def __init__(self, X , K):  # 样本矩阵X，K即要降到的维度
        self.X = X
        self.K = K
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵,即特征向量矩阵U （特征向量矩阵降维不理解）
        self.Z = []  # 样本矩阵X的降维矩阵结果Z Z=CU
        self.centrX = self._centraliazed()  # 自定义的中心化的函数
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

# 样本矩阵X中心化
    def _centraliazed(self):
        print("样本矩阵X：\n", self.X)
        centrX = []
        # 样本集的特征均值，self.X.T是self.X的转置
        mean = np.array([np.mean(attr) for attr in self.X.T])
        centrX = self.X - mean
        return centrX

# 求样本矩阵X的协方差矩阵C
    def _cov(self):
        ns = np.shape(self.centrX)[0]  # 样本集的样例总数
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)  # 求协方差矩阵。dot就是乘法，中心化矩阵的协方差矩阵公式
        print("样本矩阵X的协方差矩阵C：\n", C)
        return C

# 求样本矩阵的转换矩阵U，shape = (n,k), n是X的特征维度总数（即样本的列数），K是降维后矩阵的维度数
    def _U(self):
        a, b = np.linalg.eig(self.C)  # 把特征值赋给a,特征向量赋给b。np.linalg.eig()直接求协方差矩阵的特征值和特征向量
        ind = np.argsort(-1 * a)  # 对特征值进行降序排序/argsort是升序排序，加上-1是降序
        # 构建K阶降维的转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        return U

# 求降维矩阵Z，Z = XU, shape=(m,k), m是X的样本数（即样本的行数）
    def _Z(self):
        Z = np.dot(self.X, self.U)
        print("样本矩阵的降维矩阵：\n", Z)
        return Z

if __name__ == '__main__':
    X = np.array(
        [[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]]
    )
    K = np.shape(X)[1] - 1
    pca = CPCA(X,K)
