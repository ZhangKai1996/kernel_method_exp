import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import GridSearchCV

from dataset.load import image_data_set, dec_to_ter


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


def svc(num_iter, samples, ratio, n_classes=2, classes_1=None, classes_2=None):
    param_grid = {'C': [0.01, 0.1, 1], 'kernel': ['rbf', 'poly', 'linear'], 'gamma': [0.01, 0.1, 1]}
    param_stat = {dec_to_ter(i): 0 for i in range(27)}

    total_acc = []
    for i in range(num_iter):
        print('Iteration {}:'.format(i + 1))
        train_X, train_y, test_X, test_y = split_data_with_new_class(samples, ratio, n_classes, classes_1, classes_2)
        X, y = np.concatenate([train_X, test_X]), np.concatenate([train_y, test_y])
        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        # Model
        model = svm.SVC()
        model = GridSearchCV(model, param_grid)
        model.fit(train_X, train_y)

        best_param = model.best_params_
        print("grid.best_params_ = ", best_param, ", grid.best_score_ =", model.best_score_)
        key = "{}{}{}".format(param_grid['C'].index(best_param['C']),
                              param_grid['gamma'].index(best_param['gamma']),
                              param_grid['kernel'].index(best_param['kernel']))
        print(model.score(train_X, train_y), model.score(test_X, test_y))
        param_stat[key] += 1

        # model = svm.SVC(**best_param)
        # model.fit(train_X, train_y)
        total_acc.append(model.score(X, y))

    print('total acc:', np.mean(total_acc))
    return np.mean(total_acc)


def run():
    """Image Segmentation"""
    n_classes = 7
    n_features = 19

    # 类x vs. 类y，x != y
    # for x in range(n_classes):
    #     for y in range(x+1, n_classes):
    #         svc(num_iter=10,
    #             samples=image_data_set,
    #             ratio=0.7,
    #             n_classes=n_classes,
    #             classes_1=[x, ],
    #             classes_2=[y, ])
    #
    #         key = '{} vs. {}'.format(x, y)
    #         print(key)

    # 类c_1 vs. 类c_2,c_3,...,c_7
    for c_1 in range(n_classes):
        classes_2 = list(range(n_classes))
        classes_2.pop(c_1)

        svc(num_iter=10,
            samples=image_data_set,
            ratio=0.7,
            n_classes=n_classes,
            classes_1=[c_1, ],
            classes_2=classes_2)

        key = '{} vs. other'.format(c_1)
        print(key)
