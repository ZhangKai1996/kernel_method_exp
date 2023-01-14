import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from kfda import Kfda

from dataset.load import image_data_set, split_data, dec_to_ter


def split_data_with_new_class(samples, ratio, n_classes, c_1=None, c_2=None):
    ok = c_1 is None or c_2 is None
    if c_1 is None:
        c_1 = list(range(n_classes))
    if c_2 is None:
        c_2 = list(range(n_classes))

    train_X, train_y, test_X, test_y = [], [], [], []
    for key, values in samples.items():
        in_c1 = key in c_1
        in_c2 = key in c_2
        # print(key, c_1, c_2)
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


def svc(num_iter, samples, ratio, classes_1=None, classes_2=None, method='origin', n_comp=1):
    n_features = len(list(samples.values())[0][0])
    n_classes = len(samples)
    # print(n_features, n_classes)

    param_grid = {'C': [0.01, 0.1, 1], 'kernel': ['rbf', 'poly', 'linear'], 'gamma': [0.01, 0.1, 1]}
    param_stat = {dec_to_ter(i): 0 for i in range(27)}

    total_acc = []
    for i in range(num_iter):
        print('Iteration {}:'.format(i + 1))
        train_X, train_y, test_X, test_y = split_data_with_new_class(samples, ratio, n_classes, classes_1, classes_2)
        X_, y = np.concatenate([train_X, test_X]), np.concatenate([train_y, test_y])

        if method == 'pca':
            pca = PCA(n_components=n_comp)
            X_ = pca.fit_transform(X_)
        elif method == 'kpca':
            kpca = KernelPCA(n_components=n_comp, kernel='rbf')
            X_ = kpca.fit_transform(X_)
        elif method == 'lda':
            lda = LinearDiscriminantAnalysis(n_components=n_comp)
            X_ = lda.fit_transform(X_, y)
        elif method == 'klda':
            kfda_ = Kfda(kernel='rbf', n_components=n_comp)
            X_ = kfda_.fit_transform(X_, y)
        else:
            X_ = X_[:, :n_comp]

        split = train_y.shape[0]
        train_X, test_X = X_[:split, :], X_[split:, :]
        # print(train_X.shape, train_y.shape)

        # Model
        model = svm.SVC()
        model.fit(train_X, train_y)
        total_acc.append(model.score(X_, y))

        # break
    # print('total acc: {}'.format(np.mean(total_acc)))
    return param_stat, np.mean(total_acc)


def main(data_set, param):
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True)
    for i, (key, values) in enumerate(param.items()):
        acc_x_list = []
        mean_acc_list = []
        for j, value in enumerate(values):
            # 类1 vs. 类2
            _, mean_acc_1 = svc(num_iter=10, samples=data_set, ratio=0.7,
                                classes_1=[0, ], classes_2=[2, ],
                                method=key, n_comp=value)
            acc_x_list.append(j + 1)
            # 类1 vs. 类3
            _, mean_acc_2 = svc(num_iter=10, samples=data_set, ratio=0.7,
                                classes_1=[0, ], classes_2=[4, ],
                                method=key, n_comp=value)
            # 类1 & 2 vs. 类3
            _, mean_acc_3 = svc(num_iter=10, samples=data_set, ratio=0.7,
                                classes_1=[2, ], classes_2=[4, ],
                                method=key, n_comp=value)
            # 类1 & 2 vs. 类3
            _, mean_acc_4 = svc(num_iter=10, samples=data_set, ratio=0.7,
                                classes_1=[3, ], classes_2=[4, ],
                                method=key, n_comp=value)
            mean_acc_list.append([mean_acc_1, mean_acc_2, mean_acc_3, mean_acc_4])

        mean_acc_list = np.array(mean_acc_list)
        print('checkpoint:', key, values, mean_acc_list)
        axes[0].plot(acc_x_list, mean_acc_list[:, 0], label=key)
        axes[1].plot(acc_x_list, mean_acc_list[:, 1], label=key)
        axes[2].plot(acc_x_list, mean_acc_list[:, 2], label=key)
        axes[3].plot(acc_x_list, mean_acc_list[:, 3], label=key)

    titles = ['1 vs. 3', '1 vs. 5', '3 vs. 5', '4 vs. 5']
    for ax, title in zip(axes, titles):
        ax.set_xticks(list(range(1, 20)))
        ax.set_title(title)
        ax.legend(loc='lower right')
    fig.text(0.5, 0.04, 'The number of components', ha='center', va='center', fontsize=14)
    fig.text(0.06, 0.5, 'Accuracy', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)  # 调整两幅子图的间距
    plt.show()


def run():
    """Image Segmentation"""
    n_classes_list = [i+1 for i in range(19)]
    param_lda = {'origin': n_classes_list, 'lda': [1, ], 'klda': [1, ]}
    param_pca = {'origin': n_classes_list, 'pca': n_classes_list, 'kpca': n_classes_list}

    # main(image_data_set, param_pca)
    main(image_data_set, param_lda)
