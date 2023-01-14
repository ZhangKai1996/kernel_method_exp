import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import GridSearchCV

from dataset.load import iris_data_set, split_data, roc_and_auc, dec_to_ter


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
        print(key, c_1, c_2)
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


def svc(num_iter, samples, ratio, n_classes=2, classes_1=None, classes_2=None, filename='iris_svc'):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 10), sharey=True)
    param_grid = {'C': [0.01, 0.1, 1], 'kernel': ['rbf', 'poly', 'linear'], 'gamma': [0.01, 0.1, 1]}
    param_stat = {dec_to_ter(i): 0 for i in range(27)}
    print(param_stat)

    total_acc = []
    for i in range(num_iter):
        ax = axes[i // 5, i % 5]
        print('Iteration {}:'.format(i + 1))
        train_X, train_y, test_X, test_y = split_data_with_new_class(samples, ratio, n_classes, classes_1, classes_2)
        X, y = np.concatenate([train_X, test_X]), np.concatenate([train_y, test_y])
        print(X.shape, y.shape)

        train_acc, eval_acc, test_acc, auc_list = [], [], [], []
        acc, best_score = [], []
        for k in range(10):
            # Split dataset
            t_X, t_y, e_X, e_y = split_data(train_X, train_y, ratio=4.0 / 7.0)

            # Model
            model = svm.SVC()
            model = GridSearchCV(model, param_grid)

            # ROC and AUC
            y_score = model.fit(t_X, t_y).decision_function(X)
            auc_value = roc_and_auc(y, y_score, show=False)
            auc_list.append(auc_value)

            best_param = model.best_params_
            print("grid.best_params_ = ", best_param, ", grid.best_score_ =", model.best_score_)
            key = "{}{}{}".format(param_grid['C'].index(best_param['C']),
                                  param_grid['gamma'].index(best_param['gamma']),
                                  param_grid['kernel'].index(best_param['kernel']))
            print("grid.best_params_ = ", key)
            param_stat[key] += 1

            total_acc.append(model.score(X, y))
            accuracy_train = model.score(t_X, t_y)
            train_acc.append(accuracy_train)
            accuracy_eval = model.score(e_X, e_y)
            eval_acc.append(accuracy_eval)
            accuracy_test = model.score(test_X, test_y)
            test_acc.append(accuracy_test)
            acc.append(model.score(train_X, train_y))
            print('\t>>>', k, len(t_X), len(t_y), len(e_X), len(e_y), len(test_X), len(test_y),
                  'train_acc: {:>.6f}'.format(accuracy_train),
                  'eval_acc: {:>.6f}'.format(accuracy_eval),
                  'test_acc: {:>.6f}'.format(accuracy_test),
                  'auc: {:>.6f}'.format(auc_value))

        x_list = [i + 1 for i in range(10)]
        ax.plot(x_list, train_acc, label='train')
        ax.plot(x_list, eval_acc, label='eval')
        ax.plot(x_list, test_acc, label='test')
        ax.set_ylim(ymin=0.8, ymax=1.02)
        ax.set_xlim(xmin=1, xmax=10)
        ax.set_title(r'$\overline{auc}$' +
                     '={:>.3f}, '.format(np.mean(auc_list)) +
                     r'$\overline{acc}$' +
                     '={:>.3f}'.format(np.mean(acc)), fontsize=12)
        # break
    print('total acc: {}'.format(np.mean(total_acc)))
    lines, labels = ax.get_legend_handles_labels()
    fig.text(0.5, 0.04, 'K-Fold(K=10)', ha='center', va='center', fontsize=14)
    fig.text(0.06, 0.5, 'Accuracy', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=4)
    # fig.suptitle("Support Vector Classification", fontsize=16)
    fig.subplots_adjust(wspace=0.10, hspace=0.15)  # 调整两幅子图的间距
    fig.savefig('./scripts/' + filename + '.pdf')
    plt.show()

    fig = plt.figure()
    x_list, y_list = [], []
    for key, value in param_stat.items():
        x_list.append(key)
        y_list.append(value)
    plt.xlabel('Index')
    plt.ylabel('frequency')
    plt.plot(x_list, y_list)
    plt.show()


def run():
    """Iris"""
    # 类1 vs. 类2
    # svc(num_iter=10, samples=iris_data_set, ratio=0.7, n_classes=3, classes_1=[0, ], classes_2=[1, ])
    # 类1 vs. 类2
    # svc(num_iter=10, samples=iris_data_set, ratio=0.7, n_classes=3, classes_1=[0, ], classes_2=[2, ])
    # 类1 vs. 类2
    svc(num_iter=10, samples=iris_data_set, ratio=0.7, n_classes=3, classes_1=[0, 1, ], classes_2=[2, ])
