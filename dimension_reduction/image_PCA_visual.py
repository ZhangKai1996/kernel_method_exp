import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

from sklearn import svm
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from kfda import Kfda

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


def main_1(X, y):
    n_components = 1
    print(X.shape, y.shape)
    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(1, 3, 1)
    reds = y == 0
    blues = y == 1
    ax.scatter(y[reds]+1, X[reds, 0], c="red", s=20, edgecolor='k')
    ax.scatter(y[blues]+1, X[blues, 0], c="blue", s=20, edgecolor='k')
    ax.set_xticks(list(range(1, 3)))
    ax.set_title('Origin')

    # pca = PCA(n_components=n_components)
    # X_pca = pca.fit_transform(X)
    # pca_ratio = pca.explained_variance_ratio_
    # print(X_pca.shape)
    # print(pca_ratio, sum(pca_ratio), len(pca_ratio), X.shape)
    # ax = fig.add_subplot(1, 5, 2)
    # ax.scatter(y[reds], X_pca[reds, 0], c="red", s=20, edgecolor='k')
    # ax.scatter(y[blues], X_pca[blues, 0], c="blue", s=20, edgecolor='k')
    # ax.set_title('PCA')
    #
    # kpca = KernelPCA(n_components=n_components, kernel='rbf')
    # X_kpca = kpca.fit_transform(X)
    # print(X_kpca.shape)
    # ax = fig.add_subplot(1, 5, 3)
    # ax.scatter(y[reds], X_kpca[reds, 0], c="red", s=20, edgecolor='k')
    # ax.scatter(y[blues], X_kpca[blues, 0], c="blue", s=20, edgecolor='k')
    # ax.set_title('KPCA')

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X, y)
    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(y[reds]+1, X_lda[reds, 0], c="red", s=20, edgecolor='k')
    ax.scatter(y[blues]+1, X_lda[blues, 0], c="blue", s=20, edgecolor='k')
    ax.set_xticks(list(range(1, 3)))
    ax.set_title('LDA')

    kfda_ = Kfda(kernel='rbf', n_components=n_components)
    X_kfda = kfda_.fit_transform(X, y)
    print(X_kfda.shape)
    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(y[reds]+1, X_kfda[reds, 0], c="red", s=20, edgecolor='k')
    ax.scatter(y[blues]+1, X_kfda[blues, 0], c="blue", s=20, edgecolor='k')
    ax.set_xticks(list(range(1, 3)))
    ax.set_title('KLDA/KFDA')

    # fig.text(0.5, 0.04, 'Class', ha='center', va='center', fontsize=14)
    # fig.text(0.06, 0.5, 'Accuracy', ha='center', va='center', rotation='vertical', fontsize=14)
    plt.show()


def main_2(X, y):
    n_components = 2
    # np.random.seed(0)
    fig = plt.figure(figsize=(16, 6))

    ax = fig.add_subplot(1, 5, 1)
    reds = y == 0
    blues = y == 1
    ax.scatter(X[reds, 0], X[reds, 1], c="red", s=20, edgecolor='k')
    ax.scatter(X[blues, 0], X[blues, 1], c="blue", s=20, edgecolor='k')
    ax.set_title('Origin')

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    pca_ratio = pca.explained_variance_ratio_
    print(X_pca.shape)
    print(pca_ratio, sum(pca_ratio), len(pca_ratio), X.shape)
    ax = fig.add_subplot(1, 5, 2)
    ax.scatter(X_pca[reds, 0], X_pca[reds, 1], c="red", s=20, edgecolor='k')
    ax.scatter(X_pca[blues, 0], X_pca[blues, 1], c="blue", s=20, edgecolor='k')
    ax.set_title('PCA')

    kpca = KernelPCA(n_components=n_components, kernel='rbf')
    X_kpca = kpca.fit_transform(X)
    print(X_kpca.shape)
    ax = fig.add_subplot(1, 5, 3)
    ax.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="red", s=20, edgecolor='k')
    ax.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="blue", s=20, edgecolor='k')
    ax.set_title('KPCA')

    plt.show()


def main_3(X, y):
    n_components = 3
    # np.random.seed(0)
    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(1, 4, 1, projection='3d')
    reds = y == 0
    blues = y == 1
    greens = y == 2
    ax.scatter(X[reds, 0], X[reds, 1], X[reds, 2], c="red", s=20, edgecolor='k')
    ax.scatter(X[blues, 0], X[blues, 1], X[blues, 2], c="blue", s=20, edgecolor='k')
    ax.scatter(X[greens, 0], X[greens, 1], X[greens, 2], c="green", s=20, edgecolor='k')

    kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=1.0)
    X_kpca = kpca.fit_transform(X)
    print(X_kpca.shape)
    ax = fig.add_subplot(1, 4, 2, projection='3d')
    ax.scatter(X_kpca[reds, 0], X_kpca[reds, 1], X_kpca[reds, 2], c="red", s=20, edgecolor='k')
    ax.scatter(X_kpca[blues, 0], X_kpca[blues, 1], X_kpca[blues, 2], c="blue", s=20, edgecolor='k')
    ax.scatter(X_kpca[greens, 0], X_kpca[greens, 1], X_kpca[greens, 2], c="green", s=20, edgecolor='k')

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    pca_ratio = pca.explained_variance_ratio_
    print(X_pca.shape)
    print(pca_ratio, sum(pca_ratio), len(pca_ratio), X.shape)
    ax = fig.add_subplot(1, 4, 3, projection='3d')
    ax.scatter(X_pca[reds, 0], X_pca[reds, 1], X_pca[reds, 2], c="red", s=20, edgecolor='k')
    ax.scatter(X_pca[blues, 0], X_pca[blues, 1], X_pca[blues, 2], c="blue", s=20, edgecolor='k')
    ax.scatter(X_pca[greens, 0], X_pca[greens, 1], X_pca[greens, 2], c="green", s=20, edgecolor='k')
    plt.show()


def run():
    dataset = image_data_set
    n_classes = 7
    param_grid = {'C': [0.01, 0.1, 1], 'kernel': ['rbf', ], 'gamma': [0.01, 0.1, 1]}

    # fig, axes = plt.subplots(nrows=n_classes, ncols=1, sharey=True)
    # acc_array = []
    # for x in range(n_classes):
    #     x_list, y_list, y_list_ = [], [], []
    #     for y in range(n_classes):
    #         if x == y:
    #             x_list.append('{} vs. {}'.format(x + 1, y + 1))
    #             y_list.append([0.0, 0.0, 0.0])
    #             continue
    #
    #         acc_list = []
    #         for i in range(10):
    #             train_X, train_y, test_X, test_y = split_data_with_new_class(dataset,
    #                                                                          0.7,
    #                                                                          n_classes,
    #                                                                          c_1=[x, ],
    #                                                                          c_2=[y, ])
    #             X_, y_ = np.concatenate([train_X, test_X]), np.concatenate([train_y, test_y])
    #             model = svm.SVC()
    #             model.fit(train_X, train_y)
    #             acc_1 = model.score(X_, y_)
    #             acc_2 = model.score(train_X, train_y)
    #             acc_3 = model.score(test_X, test_y)
    #             acc_list.append([acc_1, acc_2, acc_3])
    #
    #         print(np.mean(acc_list, axis=0))
    #         x_list.append('{} vs. {}'.format(x+1, y+1))
    #         y_list.append(np.mean(acc_list, axis=0))
    #         y_list_.append(np.mean(acc_list, axis=0))
    #     acc_array.append(np.mean(y_list_, axis=0))
    #
    #     y_list = np.array(y_list)
    #     ax = axes[x]
    #     ax.plot(x_list, y_list[:, 0], label='total')
    #     ax.plot(x_list, y_list[:, 1], label='train')
    #     ax.plot(x_list, y_list[:, 2], label='test')
    #     ax.set_ylim(ymin=0.7, ymax=1.05)
    #     ax.legend(loc='lower right')
    #
    # fig.text(0.5, 0.04, 'One vs. another', ha='center', va='center', fontsize=14)
    # fig.text(0.06, 0.5, 'Accuracy', ha='center', va='center', rotation='vertical', fontsize=14)
    # fig.subplots_adjust(wspace=0.3, hspace=0.3)  # 调整两幅子图的间距
    # plt.xticks(rotation=45)
    # plt.show()
    #
    # acc_array = np.array(acc_array)
    # x_list = ['{} vs. others'.format(i+1) for i in range(n_classes)]
    # fig = plt.figure()
    # plt.plot(x_list, acc_array[:, 0], label='total')
    # plt.plot(x_list, acc_array[:, 1], label='train')
    # plt.plot(x_list, acc_array[:, 2], label='test')
    # plt.ylim(ymin=0.9)
    # plt.xlabel('one vs. others')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

    # x_list, y_list = [], []
    # for c_1 in range(n_classes):
    #     classes_2 = list(range(n_classes))
    #     classes_2.pop(c_1)
    #
    #     acc_list = []
    #     for i in range(10):
    #         train_X, train_y, test_X, test_y = split_data_with_new_class(dataset,
    #                                                                      0.7,
    #                                                                      n_classes,
    #                                                                      c_1=[c_1, ],
    #                                                                      c_2=classes_2)
    #         X_, y_ = np.concatenate([train_X, test_X]), np.concatenate([train_y, test_y])
    #         model = svm.SVC()
    #         model = GridSearchCV(model, param_grid)
    #         model.fit(train_X, train_y)
    #         acc_1 = model.score(X_, y_)
    #         acc_2 = model.score(train_X, train_y)
    #         acc_3 = model.score(test_X, test_y)
    #
    #         best_param = model.best_params_
    #         print("grid.best_params_ = ", best_param, ", grid.best_score_ =", model.best_score_)
    #         acc_list.append([acc_1, acc_2, acc_3])
    #
    #     print(np.mean(acc_list, axis=0))
    #     x_list.append('{} vs. other'.format(c_1 + 1))
    #     y_list.append(np.mean(acc_list, axis=0))
    #
    # y_list = np.array(y_list)
    # fig = plt.figure()
    # plt.plot(x_list, y_list[:, 0], label='total')
    # plt.plot(x_list, y_list[:, 1], label='train')
    # plt.plot(x_list, y_list[:, 2], label='test')
    # plt.xlabel('one vs. others')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

    train_X, train_y, test_X, test_y = split_data_with_new_class(dataset,
                                                                 0.7,
                                                                 n_classes)
    X_, y_ = np.concatenate([train_X, test_X]), np.concatenate([train_y, test_y])
    main_1(X_[:, :], y_[:])
    # main_2(X_[:, :], y_[:])
    # main_3(X_[:, :], y_[:])
