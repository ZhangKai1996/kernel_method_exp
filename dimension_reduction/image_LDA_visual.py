import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA

from dataset.load import image_data_set


def split_data_with_new_class(samples, ratio, n_classes, c_1=None, c_2=None):
    ok = c_1 is None and c_2 is None

    if c_1 is None:
        c_1 = list(range(n_classes))
    if c_2 is None:
        c_2 = list(range(n_classes))

    train_X, train_y, test_X, test_y = [], [], [], []
    for key, values in samples.items():
        in_c1 = key in c_1
        in_c2 = key in c_2
        if not in_c1 and not in_c2:
            continue

        np.random.shuffle(values)
        size = len(values)
        split = int(ratio * size)

        train_X += values[:split]
        test_X += values[split:]

        if ok:
            train_y += [key for _ in range(split)]
            test_y += [key for _ in range(split, size)]
            continue

        if in_c1:
            train_y += [-1 for _ in range(split)]
            test_y += [-1 for _ in range(split, size)]
        elif in_c2:
            train_y += [1 for _ in range(split)]
            test_y += [1 for _ in range(split, size)]
        else:
            raise NotImplementedError

    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)


def main_1(X, y, fig, n_rows, n_columns, start, n_components):
    reds = y == 0
    blues = y == 1
    greens = y == 2

    ax = fig.add_subplot(n_rows, n_columns, start + 1)
    ax.scatter(y[reds] + 1, X[reds, 0], c="red", s=20, edgecolor='k')
    ax.scatter(y[blues] + 1, X[blues, 0], c="blue", s=20, edgecolor='k')
    ax.scatter(y[greens] + 1, X[greens, 0], c="green", s=20, edgecolor='k')
    ax.set_xticks(list(range(1, 4)))
    ax.set_title('Origin(1-D)')

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    pca_ratio = pca.explained_variance_ratio_
    print(X_pca.shape)
    print(pca_ratio, sum(pca_ratio), len(pca_ratio), X.shape)
    ax = fig.add_subplot(n_rows, n_columns, start + 2)
    ax.scatter(y[reds] + 1, X_pca[reds, 0], c="red", s=20, edgecolor='k')
    ax.scatter(y[blues] + 1, X_pca[blues, 0], c="blue", s=20, edgecolor='k')
    ax.scatter(y[greens] + 1, X_pca[greens, 0], c="green", s=20, edgecolor='k')
    ax.set_xticks(list(range(1, 4)))
    ax.set_title('PCA(1-D)')

    kpca = KernelPCA(n_components=n_components, kernel='rbf')
    X_kpca = kpca.fit_transform(X)
    print(X_kpca.shape)
    ax = fig.add_subplot(n_rows, n_columns, start + 3)
    ax.scatter(y[reds] + 1, X_kpca[reds, 0], c="red", s=20, edgecolor='k')
    ax.scatter(y[blues] + 1, X_kpca[blues, 0], c="blue", s=20, edgecolor='k')
    ax.scatter(y[greens] + 1, X_kpca[greens, 0], c="green", s=20, edgecolor='k')
    ax.set_xticks(list(range(1, 4)))
    ax.set_title('KPCA(1-D)')


def main_2(X, y, fig, n_rows, n_columns, start, n_components):
    reds = y == 0
    blues = y == 1
    greens = y == 2

    ax = fig.add_subplot(n_rows, n_columns, start + 1)
    ax.scatter(X[reds, 0], X[reds, 1], c="red", s=20, edgecolor='k')
    ax.scatter(X[blues, 0], X[blues, 1], c="blue", s=20, edgecolor='k')
    ax.scatter(X[greens, 0], X[greens, 1], c="green", s=20, edgecolor='k')
    ax.set_title('Origin(2-D)')

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    pca_ratio = pca.explained_variance_ratio_
    print(X_pca.shape)
    print(pca_ratio, sum(pca_ratio), len(pca_ratio), X.shape)
    ax = fig.add_subplot(n_rows, n_columns, start + 2)
    ax.scatter(X_pca[reds, 0], X_pca[reds, 1], c="red", s=20, edgecolor='k')
    ax.scatter(X_pca[blues, 0], X_pca[blues, 1], c="blue", s=20, edgecolor='k')
    ax.scatter(X_pca[greens, 0], X_pca[greens, 1], c="green", s=20, edgecolor='k')
    ax.set_title('PCA(2-D)')

    kpca = KernelPCA(n_components=n_components, kernel='rbf')
    X_kpca = kpca.fit_transform(X)
    print(X_kpca.shape)
    ax = fig.add_subplot(n_rows, n_columns, start + 3)
    ax.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="red", s=20, edgecolor='k')
    ax.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="blue", s=20, edgecolor='k')
    ax.scatter(X_kpca[greens, 0], X_kpca[greens, 1], c="green", s=20, edgecolor='k')
    ax.set_title('KPCA(2-D)')


def main_3(X, y, fig, n_rows, n_columns, start, n_components):
    reds = y == 0
    blues = y == 1
    greens = y == 2

    ax = fig.add_subplot(n_rows, n_columns, start + 1, projection='3d')
    ax.scatter(X[reds, 0], X[reds, 1], X[reds, 2], c="red", s=20, edgecolor='k')
    ax.scatter(X[blues, 0], X[blues, 1], X[blues, 2], c="blue", s=20, edgecolor='k')
    ax.scatter(X[greens, 0], X[greens, 1], X[greens, 2], c="green", s=20, edgecolor='k')
    ax.set_title('Origin(3-D)')

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    pca_ratio = pca.explained_variance_ratio_
    print(X_pca.shape)
    print(pca_ratio, sum(pca_ratio), len(pca_ratio), X.shape)
    ax = fig.add_subplot(n_rows, n_columns, start + 2, projection='3d')
    ax.scatter(X_pca[reds, 0], X_pca[reds, 1], X_pca[reds, 2], c="red", s=20, edgecolor='k')
    ax.scatter(X_pca[blues, 0], X_pca[blues, 1], X_pca[blues, 2], c="blue", s=20, edgecolor='k')
    ax.scatter(X_pca[greens, 0], X_pca[greens, 1], X_pca[greens, 2], c="green", s=20, edgecolor='k')
    ax.set_title('KPCA(3-D)')

    kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=1.0)
    X_kpca = kpca.fit_transform(X)
    print(X_kpca.shape)
    ax = fig.add_subplot(n_rows, n_columns, start + 3, projection='3d')
    ax.scatter(X_kpca[reds, 0], X_kpca[reds, 1], X_kpca[reds, 2], c="red", s=20, edgecolor='k')
    ax.scatter(X_kpca[blues, 0], X_kpca[blues, 1], X_kpca[blues, 2], c="blue", s=20, edgecolor='k')
    ax.scatter(X_kpca[greens, 0], X_kpca[greens, 1], X_kpca[greens, 2], c="green", s=20, edgecolor='k')
    ax.set_title('PCA(3-D)')


def run():
    dataset = image_data_set
    ratio = 1.0

    n_features = len(list(dataset.values())[0][0])
    n_classes = len(dataset)
    print(n_features, n_classes)

    # np.random.seed(0)

    X_, y_, _, _ = split_data_with_new_class(dataset, ratio, n_classes)
    print(X_.shape, y_.shape)

    fig_ = plt.figure(constrained_layout=False)
    for n_components_ in range(3):
        print(n_components_)

        if n_components_ == 0:
            main_1(X_, y_, fig_, 2, 3, n_components_ * 3, n_components_ + 1)
        elif n_components_ == 1:
            main_2(X_, y_, fig_, 2, 3, n_components_ * 3, n_components_ + 1)
        else:
            raise NotImplementedError

    fig_.subplots_adjust(wspace=0.25, hspace=0.40)  # 调整两幅子图的间距
    plt.show()
