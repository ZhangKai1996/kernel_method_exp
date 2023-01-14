import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

iris_path = 'dataset/iris/iris.data'
image_path_1 = 'dataset/image segmentation/segmentation.data'
image_path_2 = 'dataset/image segmentation/segmentation.test'
iris_label_dict = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
image_label_dict = {"BRICKFACE": 0, "SKY": 1, "FOLIAGE": 2, "CEMENT": 3, "WINDOW": 4, "PATH": 5, "GRASS": 6}


def load_iris_data(filename, label_transition=None):
    samples = {}
    with open(filename, 'r', newline='') as f:
        for line in f.readlines():
            line = line.strip('\r\n').split(',')
            features = [float(v) for v in line[:-1]]
            label = label_transition[line[-1]]
            # print(features, label)
            if label in samples.keys():
                samples[label].append(features)
            else:
                samples[label] = [features]
    return samples


def load_image_data(filename, label_transition=None):
    samples = {}
    with open(filename, 'r', newline='') as f:
        for line in f.readlines():
            line = line.strip('\r\n').split(',')
            features = [float(v) for v in line[1:]]
            label = label_transition[line[0]]
            # print(label, features)
            if label in samples.keys():
                samples[label].append(features)
            else:
                samples[label] = [features]
    return samples


def split_data(samples_X, samples_y, ratio):
    size = len(samples_X)
    split = int(ratio * size)
    idx_list = list(range(size))
    np.random.shuffle(idx_list)

    a_idx_list = idx_list[:split]
    b_idx_list = idx_list[split:]
    return ([samples_X[idx, :] for idx in a_idx_list],
            [samples_y[idx] for idx in a_idx_list],
            [samples_X[idx] for idx in b_idx_list],
            [samples_y[idx] for idx in b_idx_list])


def accuracy(y_real, y_pred):
    assert len(y_real) == len(y_pred)
    return 1 - np.mean(np.power(y_real - y_pred, 2))


iris_data_set = load_iris_data(filename=iris_path, label_transition=iris_label_dict)
image_data_set_1 = load_image_data(filename=image_path_1, label_transition=image_label_dict)
image_data_set_2 = load_image_data(filename=image_path_2, label_transition=image_label_dict)
image_data_set = {}
for key, value in image_data_set_1.items():
    image_data_set[key] = value + image_data_set_2[key]


def roc_and_auc(y_test, y_score, show=True):
    # 计算ROC曲线和AUC面积的微观平均（micro-averaging）
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel(), pos_label=1)
    roc_auc = auc(fpr, tpr)

    # 绘制全部的ROC曲线
    if show:
        lw = 2
        plt.figure()
        plt.plot(fpr, tpr,
                 label='ROC curve (area = {0:0.3f})'.format(roc_auc),
                 color='deeppink', linestyle=':', linewidth=lw)

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()
    return roc_auc


def dec_to_ter(num, length=3):
    l = []
    if num < 0:
        return "- " + dec_to_ter(abs(num))  # 负数先转为正数，再调用函数主体
    else:
        while True:
            num, reminder = divmod(num, 3)  # 算除法求除数和余数
            l.append(str(reminder))  # 将余数存入字符串
            if num == 0:
                if length - len(l) > 0:
                    l += ['0' for _ in range(length-len(l))]
                return "".join(l[::-1])
