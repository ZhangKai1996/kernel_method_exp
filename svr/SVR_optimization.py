import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


def main():
    # 获得样本数据
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y_ = (2 * np.sin(X) + np.sin(2 * X)).ravel()
    print(X.shape, y_.shape)
    model_color = ['m', 'c', 'g']
    lw = 2

    for idx, k_label in enumerate([('rbf', 'RBF'), ('linear', 'Linear'), ('poly', 'Polynomial')]):
        fig_rbf, axes_rbf = plt.subplots(nrows=3, ncols=3, figsize=(16, 9), sharey=True)
        for i, sigma in enumerate([0.1, 0.5, 1.0]):
            for j, epsilon in enumerate([0.01, 0.05, 1.0]):
                # 在标签中增加噪音
                y = y_ + np.random.normal(0.0, sigma, y_.shape)

                # 拟合回归模型
                svr_rbf = SVR(kernel=k_label[0], epsilon=epsilon)
                svr_rbf.fit(X, y)
                y_pred = svr_rbf.predict(X)

                # 结果可视化
                ax = axes_rbf[i, j]
                ax.plot(X, y_pred, color=model_color[idx], lw=lw, label='{} model'.format(k_label[1]))
                ax.scatter(X[svr_rbf.support_],
                           y[svr_rbf.support_],
                           facecolor="none",
                           edgecolor=model_color[idx], s=50,
                           label='{} support vectors'.format(k_label[1]))
                ax.scatter(X[np.setdiff1d(np.arange(len(X)), svr_rbf.support_)],
                           y[np.setdiff1d(np.arange(len(X)), svr_rbf.support_)],
                           facecolor="none", edgecolor="k", s=50,
                           label='other training data')
                ax.set_title(
                    r'$\sigma$''={:>.2f}'.format(sigma) + ', ' + r'$\epsilon$''={:>.2f}'.format(epsilon) +
                    '\n mse: {:>.2f}, r2_score: {:>.3f}'.format(mean_squared_error(y, y_pred), r2_score(y, y_pred))
                )
        lines, labels = ax.get_legend_handles_labels()
        fig_rbf.legend(lines, labels, ncol=3, bbox_to_anchor=(0.7, 0.98), fancybox=True, shadow=True)
        # fig_rbf.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.75, 0.96),  fancybox=True,
        # shadow=True)
        fig_rbf.text(0.5, 0.05, r'$x$', ha='center', va='center', fontsize=16)
        fig_rbf.text(0.07, 0.5, r'$y$', ha='center', va='center', rotation='vertical', fontsize=16)
        fig_rbf.subplots_adjust(wspace=0.10, hspace=0.40)  # 调整两幅子图的间距
        # fig_rbf.suptitle("Support Vector Regression ({})".format(k_label[0]), fontsize=16)
        fig_rbf.savefig('SVR_{}.pdf'.format(k_label[0]))
        plt.show()


def run():
    # 获得样本数据
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y_ = (2 * np.sin(X) + np.sin(2 * X)).ravel()
    print(X.shape, y_.shape)

    sigmas = np.linspace(0.0, 1.01, 100)
    epsilons = np.linspace(0.0, 1.01, 100)

    for idx, k_label in enumerate([('rbf', 'RBF'), ('linear', 'Linear'), ('poly', 'Polynomial')]):
        fig = plt.figure(figsize=(16, 7))
        f_mse, f_r2 = [], []
        for sigma in sigmas:
            for epsilon in epsilons:
                # 在标签中增加噪音
                y = y_ + np.random.normal(0.0, sigma, y_.shape)

                # 拟合回归模型
                svr_rbf = SVR(kernel=k_label[0], epsilon=epsilon)
                svr_rbf.fit(X, y)
                y_pred = svr_rbf.predict(X)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                f_mse.append(mse)
                f_r2.append(r2)

        ax = fig.add_subplot(1, 2, 1)
        x, y = np.meshgrid(sigmas, epsilons)
        f_mse = np.array(f_mse).reshape(x.shape[0], -1)
        c_mse = ax.contourf(x, y, f_mse, cmap='Reds')
        ax.set_title(r'$MSE$'+'({})'.format(k_label[1]), fontsize=18)
        fig.colorbar(c_mse, ax=ax, shrink=0.75)

        ax = fig.add_subplot(1, 2, 2)
        f_r2 = np.array(f_r2).reshape(x.shape[0], -1)
        c_r2 = ax.contourf(x, y, f_r2, cmap='Blues')
        ax.set_title(r'$R^2$'+'({})'.format(k_label[1]), fontsize=18)
        fig.colorbar(c_r2, ax=ax, shrink=0.75)

        fig.text(0.5, 0.05, r'$\sigma$', ha='center', va='center', fontsize=16)
        fig.text(0.07, 0.5, r'$\epsilon$', ha='center', va='center', rotation='vertical', fontsize=16)
        plt.show()
