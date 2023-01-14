import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# 导入数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
print(y, y.shape)

# 将输出二值化
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# 增加一些噪音特征让问题变得更难一些
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# 打乱并分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

# 学习，预测
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# 计算每个类别的ROC曲线和AUC面积
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算ROC曲线和AUC面积的微观平均（micro-averaging）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

lw = 2

# 首先收集所有的假正率
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# 然后在此点内插所有ROC曲线
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# 最终计算平均和ROC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# 绘制全部的ROC曲线
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


# y_prob = classifier.predict_proba(X_test)
#
# macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
#                                   average="macro")
# weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
#                                      average="weighted")
# macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
#                                   average="macro")
# weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
#                                      average="weighted")
# print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
#       "(weighted by prevalence)"
#       .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
# print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
#       "(weighted by prevalence)"
#       .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))
