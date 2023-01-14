"""
重新编写PCA/KPCA的代码
"""
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.decomposition import KernelPCA
from scipy.spatial.distance import pdist, squareform


class PCA:
    def __init__(self):
        self.eigen_values = None
        self.eigen_vectors = None
        self.k = 2

    def standardize(self, X):
        X_std = np.zeros(X.shape)
        mean = X.mean(axis=0)
        std = X.std(axis=0)

        for col in range(np.shape(X)[1]):
            if std[col]:
                X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
        return X_std

    def cov_matrix(self, X, Y=np.empty((0, 0))):
        if not Y.any():
            Y = X

        n_samples = np.shape(X)[0]
        covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))
        return np.array(covariance_matrix, dtype=float)

    def fit_transform(self, X):
        X_central = self.standardize(X)

        # 协方差矩阵
        covariance = self.cov_matrix(X_central)
        # 求解特征值和特征向量
        self.eigen_values, self.eigen_vectors = np.linalg.eig(covariance)
        # 将特征值从大到小进行排序，注意特征向量是按列排的，即eigenvectors第k列是eigenvalues中第k个特征值对应的特征向量
        idx = self.eigen_values.argsort()[::-1]
        eigenvalues = self.eigen_values[idx][:self.k]
        eigenvectors = self.eigen_vectors[:, idx][:, :self.k]
        # 将原始数据集 X 映射到低维空间
        return X.dot(eigenvectors)


class KPCA:
    def __init__(self, n_components=2, kernel='rbf'):
        self.n_components = n_components
        if kernel == 'rbf':
            self.kernel = self.rbf
        elif kernel == 'linear':
            self.kernel = self.linear
        elif kernel == 'sigmoid':
            self.kernel = self.sigmoid
        else:
            raise NotImplementedError

    def sigmoid(self, x, coef=0.05):
        x = np.dot(x, x.T)
        return np.tanh(coef * x + 1)

    def linear(self, x):
        x = np.dot(x, x.T)
        return x

    def rbf(self, x, gamma=15):
        sq_dists = pdist(x, 'sqeuclidean')
        mat_sq_dists = squareform(sq_dists)
        return np.exp(-gamma * mat_sq_dists)

    def fit_transform(self, X):
        n_components = self.n_components
        kernel = self.kernel

        K = kernel(X)
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        eig_values, eig_vector = np.linalg.eig(K)
        idx = eig_values.argsort()[::-1]
        eigval = eig_values[idx][:n_components]
        eigvector = eig_vector[:, idx][:, :n_components]
        eigval = eigval ** (1 / 2)
        vi = eigvector / eigval.reshape(-1, n_components)
        data_n = np.dot(K, vi)
        return data_n


if __name__ == "__main__":
    dataset = load_iris()
    X = dataset.data
    y = dataset.target

    model = KPCA(n_components=2, kernel='rbf')
    X_new_1 = model.fit_transform(X)

    sklearn_kpca = KernelPCA(n_components=2, kernel="rbf", gamma=15)
    X_new_2 = sklearn_kpca.fit_transform(X)

    plt.figure()
    plt.subplot(121)
    plt.title("KPCA by sklearn")
    plt.scatter(X_new_2[:, 0], X_new_2[:, 1], c=y)

    plt.subplot(122)
    plt.title("My KPCA")
    plt.scatter(X_new_1[:, 0], X_new_1[:, 1], c=y)
    plt.show()
