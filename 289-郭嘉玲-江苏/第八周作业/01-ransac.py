import scipy as sp
import numpy as np
import scipy.linalg as sl

def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    iteration = 0
    bestfit = None
    bestfit = np.inf  # 初始化最佳拟合模型参数为无穷大
    best_inlier_idxs = None

    while iteration < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])  # 随机划分数据
        maybe_inliers = data[maybe_idxs, :]
        test_points = data[test_idxs]

        maybemodel = model.fit(maybe_inliers)
        test_err = model.get_error(test_points, maybemodel)
        print('test_err = ', test_err < t)
        also_idxs = test_idxs[test_err < t]
        print('also_idxs = ', also_idxs)
        also_inliers = data[also_idxs, :]

        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d: len(alsoinliers = %d' % (iteration, len(also_inliers)))

        if len(also_inliers) > d:
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)

            if thiserr < besterr:
                bestfit = bettermodel  # 更新最佳拟合模型
                besterr = thiserr  # 更新最佳误差
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新最佳内点索引，将新内点加入
        iteration += 1

        if bestfit is None:
            raise ValueError("did't meet fit acceptance criteria")
        if return_all:
            return bestfit, {'inliears': best_inlier_idxs}
        else:
            return bestfit

def random_partition(n, n_data):
    all_idx = np.arange(n_data)
    np.random.shuffle(all_idx)
    idxs1 = all_idx[:n]
    idxs2 = all_idx[n:]
    return idxs1, idxs2

# 最小二乘线性模型，用于ransac算法的输入模型
class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 输入矩阵
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 输出矩阵
        x, resids, rank, s = sl.lstsq(A, B)  # 最小二乘法拟合模型
        return x

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = np.dot(A, model)
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point

def test():
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20 * np.random.random((n_samples, n_inputs))
    perferct_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    B_exact = np.dot(A_exact, perferct_fit)
    # 加入高斯噪声
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    if 1:
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])
        np.random.shuffle(all_idxs)
        outlier_idxs = all_idxs[:n_outliers]
        A_noisy[outlier_idxs] = 20*np.random.random((n_outliers, n_inputs))
        B_noisy[outlier_idxs] = 50* 
