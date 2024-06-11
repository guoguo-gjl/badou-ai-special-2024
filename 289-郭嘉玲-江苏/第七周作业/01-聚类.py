# 01-层次聚类
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
Z = linkage(X, 'ward')
fig = plt.figure(figsize=(5, 3))
T = dendrogram(Z)
print(Z)
plt.show()

# 01-密度聚类
from sklearn import datasets
from sklearn.cluster import DBSCAN
iris = datasets.load_iris()
X = iris.data[:, :4]
dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_
print(label_pred)
X0 = X[label_pred == 0]
X1 = X[label_pred == 1]
X2 = X[label_pred == 2]
X3 = X[label_pred == -1]
plt.scatter(X0[:, 0], X0[:, 1], c='red', marker='o', label='label0')
plt.scatter(X1[:, 0], X1[:, 1], c='green', marker='*', label='label1')
plt.scatter(X2[:, 0], X2[:, 1], c='blue', marker='+', label='label2')
plt.scatter(X3[:, 0], X3[:, 1], c='yellow', marker='+', label='label3')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()