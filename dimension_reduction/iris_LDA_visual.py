import numpy as np
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from kfda import Kfda

from dataset.load import iris_data_set


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


def main_1(fig, n_rows, n_columns, start, n_components):
    reds = y == 0
    blues = y == 1
    greens = y == 2

    ax = fig.add_subplot(n_rows, n_columns, start+1)
    ax.scatter(y[reds]+1, X[reds, 0], c="red", s=20, edgecolor='k')
    ax.axhline(y=min(X[reds, 0]), c='red')
    ax.axhline(y=max(X[reds, 0]), c='red')
    ax.scatter(y[blues]+1, X[blues, 0], c="blue", s=20, edgecolor='k')
    ax.axhline(y=min(X[blues, 0]), c='blue')
    ax.axhline(y=max(X[blues, 0]), c='blue')
    ax.scatter(y[greens]+1, X[greens, 0], c="green", s=20, edgecolor='k')
    ax.axhline(y=min(X[greens, 0]), c='green')
    ax.axhline(y=max(X[greens, 0]), c='green')
    ax.set_xticks(list(range(1, 4)))
    ax.set_title('Origin(1-D)')

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X, y)
    ax = fig.add_subplot(n_rows, n_columns, start+2)
    ax.scatter(y[reds]+1, X_lda[reds, 0], c="red", s=20, edgecolor='k')
    ax.axhline(y=min(X_lda[reds, 0]), c='red')
    ax.axhline(y=max(X_lda[reds, 0]), c='red')
    ax.scatter(y[blues]+1, X_lda[blues, 0], c="blue", s=20, edgecolor='k')
    ax.axhline(y=min(X_lda[blues, 0]), c='blue')
    ax.axhline(y=max(X_lda[blues, 0]), c='blue')
    ax.scatter(y[greens]+1, X_lda[greens, 0], c="green", s=20, edgecolor='k')
    ax.axhline(y=min(X_lda[greens, 0]), c='green')
    ax.axhline(y=max(X_lda[greens, 0]), c='green')
    ax.set_xticks(list(range(1, 4)))
    ax.set_title('LDA(1-D)')

    kfda_ = Kfda(kernel='rbf', n_components=n_components, gamma=0.01)
    X_kfda = kfda_.fit_transform(X, y)
    print(X_kfda.shape)
    ax = fig.add_subplot(n_rows, n_columns, start+3)
    ax.scatter(y[reds]+1, X_kfda[reds, 0], c="red", s=20, edgecolor='k')
    ax.axhline(y=min(X_kfda[reds, 0]), c='red')
    ax.axhline(y=max(X_kfda[reds, 0]), c='red')
    ax.scatter(y[blues]+1, X_kfda[blues, 0], c="blue", s=20, edgecolor='k')
    ax.axhline(y=min(X_kfda[blues, 0]), c='blue')
    ax.axhline(y=max(X_kfda[blues, 0]), c='blue')
    ax.text(2, -8, '[{:>.2f},{:>.2f}]'.format(min(X_kfda[blues, 0]),
                                              max(X_kfda[blues, 0])), ha='center', va='center')
    ax.scatter(y[greens]+1, X_kfda[greens, 0], c="green", s=20, edgecolor='k')
    ax.axhline(y=min(X_kfda[greens, 0]), c='green')
    ax.axhline(y=max(X_kfda[greens, 0]), c='green')
    ax.text(3, -8, '[{:>.2f},{:>.2f}]'.format(min(X_kfda[greens, 0]),
                                              max(X_kfda[greens, 0])), ha='center', va='center')
    ax.set_xticks(list(range(1, 4)))
    ax.set_title('KLDA/KFDA(1-D)')


def main_2(fig, n_rows, n_columns, start, n_components):
    reds = y == 0
    blues = y == 1
    greens = y == 2

    ax = fig.add_subplot(n_rows, n_columns, start+1)
    ax.scatter(X[reds, 0], X[reds, 1], c="red", s=20, edgecolor='k')
    ax.scatter(X[blues, 0], X[blues, 1], c="blue", s=20, edgecolor='k')
    ax.scatter(X[greens, 0], X[greens, 1], c="green", s=20, edgecolor='k')
    ax.set_title('Origin(2-D)')

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X, y)
    ax = fig.add_subplot(n_rows, n_columns, start+2)
    ax.scatter(X_lda[reds, 0], X_lda[reds, 1], c="red", s=20, edgecolor='k')
    ax.scatter(X_lda[blues, 0], X_lda[blues, 1], c="blue", s=20, edgecolor='k')
    ax.scatter(X_lda[greens, 0], X_lda[greens, 1], c="green", s=20, edgecolor='k')
    ax.set_title('LDA(2-D)')

    kfda_ = Kfda(kernel='rbf', n_components=n_components, gamma=1)
    X_kfda = kfda_.fit_transform(X, y)
    print(X_kfda.shape)
    ax = fig.add_subplot(n_rows, n_columns, start+3)
    ax.scatter(X_kfda[reds, 0], X_kfda[reds, 1], c="red", s=20, edgecolor='k')
    ax.scatter(X_kfda[blues, 0], X_kfda[blues, 1], c="blue", s=20, edgecolor='k')
    ax.scatter(X_kfda[greens, 0], X_kfda[greens, 1], c="green", s=20, edgecolor='k')
    ax.set_title('KLDA/KFDA(2-D)')


def run():
    dataset = iris_data_set
    ratio = 1.0

    n_features = len(list(dataset.values())[0][0])
    n_classes = len(dataset)
    print(n_features, n_classes)

    # np.random.seed(0)

    X, y, _, _ = split_data_with_new_class(dataset, ratio, n_classes)
    print(X.shape, y.shape)

    fig_ = plt.figure(constrained_layout=False)
    for n_components_ in range(2):
        print(n_components_)

        if n_components_ == 0:
            main_1(fig_, 2, 3, n_components_*3, n_components_+1)
        elif n_components_ == 1:
            main_2(fig_, 2, 3, n_components_*3, n_components_+1)
        else:
            raise NotImplementedError

    fig_.subplots_adjust(wspace=0.25, hspace=0.40)  # 调整两幅子图的间距
    plt.show()
