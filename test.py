import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from sklearn import model_selection, metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_projective_point(point, line):
    a = point[0]
    b = point[1]
    k = line[0]
    t = line[1]

    if k == 0:
        return [a, t]
    elif k == np.inf:
        return [0, b]
    x = (a + k * b - k * t) / (k * k + 1)
    y = k * x + t
    return [x, y]


if __name__ == '__main__':
    dataset = np.loadtxt('./dataset/watermelon_3a.csv', delimiter=',')
    X = dataset[:, 1:3]
    y = dataset[:, 3]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)

    axes[0].scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='red', s=100, label='bad')
    axes[0].scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='blue', s=100, label='good')

    m, n = np.shape(X)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)
    # training
    lda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X_train, y_train)

    # testing
    y_pred = lda_model.predict(X_test)
    print('w:\n', lda_model.coef_)
    print('b:\n', lda_model.intercept_)
    print('Accuracy:\n', metrics.accuracy_score(y_test, y_pred))
    print('Confusion Matrix:\n', metrics.confusion_matrix(y_test, y_pred))
    print('Classification Report:\n', metrics.classification_report(y_test, y_pred))

    h = 0.001
    x0_min, x0_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x1_min, x1_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    x0, x1 = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))
    # your model's prediction (classification) function
    z = lda_model.predict(np.c_[x0.ravel(), x1.ravel()])

    # Put the result into a color plot
    z = z.reshape(x0.shape)
    axes[1].contourf(x0, x1, z, cmap=pl.cm.Paired)
    axes[1].scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='red', s=100, label='bad')
    axes[1].scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='blue', s=100, label='good')

    w = lda_model.coef_.reshape(n, 1)
    p0_x = -X[:, 0].max()
    p0_y = (w[1, 0] / w[0, 0]) * p0_x
    p1_x = X[:, 0].max()
    p1_y = (w[1, 0] / w[0, 0]) * p1_x

    axes[2].scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='red', s=100, label='bad')
    axes[2].scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='blue', s=100, label='good')
    axes[2].plot([p0_x, p1_x], [p0_y, p1_y])

    m, n = np.shape(X)
    for i in range(m):
        x_p = get_projective_point([X[i, 0], X[i, 1]], [w[1, 0] / w[0, 0], 0])
        if y[i] == 0:
            plt.plot(x_p[0], x_p[1], 'ro', markersize=5)
        if y[i] == 1:
            plt.plot(x_p[0], x_p[1], 'bo', markersize=5)
        axes[2].plot([x_p[0], X[i, 0]], [x_p[1], X[i, 1]], 'c--', linewidth=0.3)

    fig.suptitle('watermelon_3a')
    fig.text(0.5, 0.04, 'density', ha='center', va='center', fontsize=14)
    fig.text(0.06, 0.5, 'ratio of sugar', ha='center', va='center', rotation='vertical', fontsize=14)
    plt.xlim(-0.2, 1)
    plt.ylim(-0.2, 0.5)
    plt.legend(loc='lower right')
    plt.show()
