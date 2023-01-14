"""
重新编写二分类下的LDA/KLDA代码
"""
import numpy as np
from numpy import linalg

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


def plot_decision_border(X, y, classifier, resolution=0.0001):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
    plt.show()


class LDA:
    def __init__(self, n_components=1):
        self.n_components = n_components

    def fit_transform(self, X, y):
        n_components = self.n_components

        clusters = np.unique(y)
        if n_components > len(clusters) - 1:
            exit(0)

        S_w = np.zeros((X.shape[1], X.shape[1]))
        for i in clusters:
            x_i = X[y == i]
            x_i = x_i - x_i.mean(0)
            S_w += np.mat(x_i).T * np.mat(x_i)

        S_b = np.zeros((X.shape[1], X.shape[1]))
        u = X.mean(0)
        for i in clusters:
            N_i = X[y == i].shape[0]
            u_i = X[y == i].mean(0)
            S_b += N_i * np.mat(u_i - u).T * np.mat(u_i - u)

        S = np.linalg.inv(S_w) * S_b
        eigenvalues, eigenvectors = np.linalg.eig(S)  # 求特征值，特征向量
        index = np.argsort(eigenvalues)[:(-n_components - 1):-1]
        return np.dot(X, eigenvectors[:, index])


class KLDA:
    def __init__(self, C=0.01, gamma=0.01):
        self.C = C
        self.gamma = gamma

    # 标准化
    def standardization(self, X):
        return StandardScaler().fit_transform(X)

    # 计算M矩阵
    def M_matrix(self, X, y):
        gamma = self.gamma
        c = self.C

        K_m = []
        size = len([i for i in y if i == c])
        for row in X:
            K_1 = 0.0
            for c_row in X[y == c]:
                K_1 += np.exp(-gamma * (np.sum((row - c_row) ** 2)))
            K_m.append(K_1 / size)
        return np.array(K_m)

    # 计算N矩阵
    def N_matrix(self, X, y):
        gamma = self.gamma
        c = self.C

        n = X.shape[0]
        c_len = len([i for i in y if i == c])
        I = np.eye(c_len)
        I_n = np.eye(n)
        I_c = np.ones((c_len, c_len)) / c_len
        K_one = np.zeros((X.shape[0], c_len))

        for i in range(n):
            K_one[i, :] = np.array([np.exp(-gamma * np.sum((X[i] - c_row) ** 2)) for c_row in X[y == c]])
        K_n = K_one.dot(I - I_c).dot(K_one.T)
        return K_n

    # 计算映射后的样本数据
    def project(self, X_new, X, alphas=[]):
        gamma = self.gamma

        N = X.shape[0]
        X_proj = np.zeros((N, 1))
        for i in range(len(X_new)):
            k = np.exp(-gamma * np.array([np.sum((X_new[i] - row) ** 2) for row in X]))
            X_proj[i, 0] = np.real(k[np.newaxis, :].dot(alphas))
        return X_proj

    def fit_transform(self, X, y):
        N = X.shape[0]

        # 求判别式广义特征值和特征向量
        K_m0 = self.M_matrix(X, y)
        K_m1 = self.M_matrix(X, y)
        K_m = (K_m0 - K_m1)[:, np.newaxis].dot((K_m0 - K_m1)[np.newaxis, :])

        K_n = np.zeros((N, N))
        for _ in np.unique(y):
            K_n += self.N_matrix(X, y)

        eig_values, eig_vectors = np.linalg.eig(np.linalg.inv(K_n).dot(K_m))
        eigen_pairs = [(np.abs(eig_values[i]), eig_vectors[:, i])
                       for i in range(len(eig_values))]
        eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
        alpha_1 = eigen_pairs[0][1][:, np.newaxis]
        alpha_2 = eigen_pairs[1][1][:, np.newaxis]

        # 新样本点
        X_new = np.zeros((N, 2))
        X_new[:, 0][:, np.newaxis] = self.project(X[:, :], X, alpha_1)
        X_new[:, 1][:, np.newaxis] = self.project(X[:, :], X, alpha_2)

        return X_new


if __name__ == '__main__':
    """测试KLDA"""
    from sklearn.datasets import make_moons
    from sklearn.svm import SVC

    X, y = make_moons(n_samples=200, noise=0.05, random_state=1)
    print(X.shape, y.shape)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='s', label='1')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', label='2')
    plt.legend(loc='upper right')
    plt.show()

    # 使用SVC对老样本进行分类
    model = SVC()
    model.fit(X, y)
    plot_decision_border(X, y, model, resolution=0.02)

    klda = KLDA()
    X_new = klda.fit_transform(X, y)
    plt.scatter(X_new[y == 0, 0], X_new[y == 0, 1], c='red', marker='s', label='$1^\'$')
    plt.scatter(X_new[y == 1, 0], X_new[y == 1, 1], c='blue', marker='o', label='$2^\'$')
    plt.legend(loc='upper right')
    plt.show()

    # 使用SVC对新样本进行分类
    model = SVC()
    model.fit(X_new, y)
    plot_decision_border(X_new, y, model, resolution=0.02)

    """测试LDA"""
    iris = load_iris()
    X = iris.data
    y = iris.target

    plt.figure()
    plt.subplot(121)
    X_new_1 = LDA(n_components=2).fit_transform(X, y)
    plt.scatter(X_new_1[:, 0], X_new_1[:, 1], c=y)
    plt.title("My LDA")

    plt.subplot(122)
    X_new_2 = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
    plt.scatter(X_new_2[:, 0], X_new_2[:, 1], c=y)
    plt.title("LDA by sklearn")

    plt.show()
