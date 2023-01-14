import numpy as np
import matplotlib.pyplot as plt

from dataset.load import iris_data_set, image_data_set


def split_data_with_new_class(samples, ratio, n_classes=2, c_1=None, c_2=None):
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


def plot_origin(samples, ratio=0.75, n_classes=2):
    # np.random.seed(0)
    X, y, test_X, test_y = split_data_with_new_class(samples, ratio, n_classes)
    print(X.shape, y.shape)

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    reds = y == 0
    blues = y == 1
    greens = y == 2
    ax.scatter(X[reds, 0], X[reds, 1], X[reds, 2], c="red", s=20, edgecolor='k')
    ax.scatter(X[blues, 0], X[blues, 1], X[blues, 2], c="blue", s=20, edgecolor='k')
    ax.scatter(X[greens, 0], X[greens, 1], X[greens, 2], c="green", s=20, edgecolor='k')
    ax.set_zlabel('petal length', fontdict={'size': 13, 'color': 'red'})
    ax.set_ylabel('sepal width', fontdict={'size': 13, 'color': 'red'})
    ax.set_xlabel('sepal length', fontdict={'size': 13, 'color': 'red'})

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.scatter(X[reds, 0], X[reds, 1], X[reds, 3], c="red", s=20, edgecolor='k')
    ax.scatter(X[blues, 0], X[blues, 1], X[blues, 3], c="blue", s=20, edgecolor='k')
    ax.scatter(X[greens, 0], X[greens, 1], X[greens, 3], c="green", s=20, edgecolor='k')
    ax.set_zlabel('petal width', fontdict={'size': 13, 'color': 'red'})
    ax.set_ylabel('sepal width', fontdict={'size': 13, 'color': 'red'})
    ax.set_xlabel('sepal length', fontdict={'size': 13, 'color': 'red'})

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.scatter(X[reds, 0], X[reds, 2], X[reds, 3], c="red", s=20, edgecolor='k')
    ax.scatter(X[blues, 0], X[blues, 2], X[blues, 3], c="blue", s=20, edgecolor='k')
    ax.scatter(X[greens, 0], X[greens, 2], X[greens, 3], c="green", s=20, edgecolor='k')
    ax.set_zlabel('petal width', fontdict={'size': 13, 'color': 'red'})
    ax.set_ylabel('petal length', fontdict={'size': 13, 'color': 'red'})
    ax.set_xlabel('sepal length', fontdict={'size': 13, 'color': 'red'})

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.scatter(X[reds, 1], X[reds, 2], X[reds, 3], c="red", s=20, edgecolor='k', label='(class 1)Setosa')
    ax.scatter(X[blues, 1], X[blues, 2], X[blues, 3], c="blue", s=20, edgecolor='k', label='(class 2)Versicolour')
    ax.scatter(X[greens, 1], X[greens, 2], X[greens, 3], c="green", s=20, edgecolor='k', label='(class 3)Virginica')
    ax.set_zlabel('petal width', fontdict={'size': 13, 'color': 'red'})
    ax.set_ylabel('petal length', fontdict={'size': 13, 'color': 'red'})
    ax.set_xlabel('sepal width', fontdict={'size': 13, 'color': 'red'})

    # lines, labels = ax.get_legend_handles_labels()
    # fig.legend(lines, labels, fancybox=True, shadow=True)
    fig.legend(loc='upper center', bbox_to_anchor=(0.53, 0.5), fancybox=True, shadow=True)
    fig.subplots_adjust(wspace=0.10, hspace=0.10)  # 调整两幅子图的间距
    # fig.suptitle("The visualization of Iris Dataset (3D)", fontsize=16)
    plt.show()


def plot_pieces(samples, ratio, n_classes, classes_1=None, classes_2=None):
    # np.random.seed(0)
    X, y, test_X, test_y = split_data_with_new_class(samples, ratio, n_classes, classes_1, classes_2)
    print(X.shape, y.shape)

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    reds = y == 0
    blues = y == 1
    greens = y == 2
    ax.scatter(X[reds, 0], X[reds, 1], X[reds, 2], c="red", s=20, edgecolor='k')
    ax.scatter(X[blues, 0], X[blues, 1], X[blues, 2], c="blue", s=20, edgecolor='k')
    ax.scatter(X[greens, 0], X[greens, 1], X[greens, 2], c="green", s=20, edgecolor='k')
    ax.set_zlabel('petal length', fontdict={'size': 13, 'color': 'red'})
    ax.set_ylabel('sepal width', fontdict={'size': 13, 'color': 'red'})
    ax.set_xlabel('sepal length', fontdict={'size': 13, 'color': 'red'})

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.scatter(X[reds, 0], X[reds, 1], X[reds, 3], c="red", s=20, edgecolor='k')
    ax.scatter(X[blues, 0], X[blues, 1], X[blues, 3], c="blue", s=20, edgecolor='k')
    ax.scatter(X[greens, 0], X[greens, 1], X[greens, 3], c="green", s=20, edgecolor='k')
    ax.set_zlabel('petal width', fontdict={'size': 13, 'color': 'red'})
    ax.set_ylabel('sepal width', fontdict={'size': 13, 'color': 'red'})
    ax.set_xlabel('sepal length', fontdict={'size': 13, 'color': 'red'})

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.scatter(X[reds, 0], X[reds, 2], X[reds, 3], c="red", s=20, edgecolor='k')
    ax.scatter(X[blues, 0], X[blues, 2], X[blues, 3], c="blue", s=20, edgecolor='k')
    ax.scatter(X[greens, 0], X[greens, 2], X[greens, 3], c="green", s=20, edgecolor='k')
    ax.set_zlabel('petal width', fontdict={'size': 13, 'color': 'red'})
    ax.set_ylabel('petal length', fontdict={'size': 13, 'color': 'red'})
    ax.set_xlabel('sepal length', fontdict={'size': 13, 'color': 'red'})

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.scatter(X[reds, 1], X[reds, 2], X[reds, 3], c="red", s=20, edgecolor='k', label='(class 1)Setosa')
    ax.scatter(X[blues, 1], X[blues, 2], X[blues, 3], c="blue", s=20, edgecolor='k', label='(class 2)Versicolour')
    ax.scatter(X[greens, 1], X[greens, 2], X[greens, 3], c="green", s=20, edgecolor='k', label='(class 3)Virginica')
    ax.set_zlabel('petal width', fontdict={'size': 13, 'color': 'red'})
    ax.set_ylabel('petal length', fontdict={'size': 13, 'color': 'red'})
    ax.set_xlabel('sepal width', fontdict={'size': 13, 'color': 'red'})

    # lines, labels = ax.get_legend_handles_labels()
    # fig.legend(lines, labels, fancybox=True, shadow=True)
    fig.legend(loc='upper center', bbox_to_anchor=(0.53, 0.5), fancybox=True, shadow=True)
    fig.subplots_adjust(wspace=0.10, hspace=0.10)  # 调整两幅子图的间距
    # fig.suptitle("The visualization of Iris Dataset (3D)", fontsize=16)
    plt.show()


if __name__ == '__main__':
    plot_origin(samples=iris_data_set, ratio=1.0, n_classes=3)
    plot_origin(samples=image_data_set, ratio=0.7, n_classes=7)
    plot_pieces(samples=iris_data_set, ratio=0.7, n_classes=3, classes_1=[0, 1, ], classes_2=[2, ])
    plot_pieces(samples=image_data_set, ratio=0.7, n_classes=7, classes_1=[0, 1, ], classes_2=[2, ])
