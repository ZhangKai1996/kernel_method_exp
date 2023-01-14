from dimension_reduction import iris_PCA_visual, image_PCA_visual, image_LDA_visual
from svc import image_SVC_pca_lda
import numpy as np
import matplotlib.pyplot as plt

# iris_PCA_visual.run()
image_PCA_visual.run()

# image_SVC_pca_lda.run()
# image_LDA_visual.run()


def read_and_visual(filename):
    with open(filename, 'r', newline='') as f:
        lines = f.readlines()
        x_list = [int(v) for v in lines[0].strip('\r\n').split(',')]
        print(x_list)

        lst, ret, key = [], {}, None
        for line in lines[1:]:
            line = line.strip('\r\n')

            if '0' not in line:
                if key is not None:
                    ret[key] = np.array(lst)
                    lst = []
                key = line
                print(line, key)
                continue

            lst.append([float(v) for v in line.split(',')])

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True)
    for i, (key, values) in enumerate(ret.items()):
        print('checkpoint:', key, values.shape)
        axes[0].plot(x_list, values[:, 0], label=key)
        axes[1].plot(x_list, values[:, 1], label=key)
        axes[2].plot(x_list, values[:, 2], label=key)
        axes[3].plot(x_list, values[:, 3], label=key)

    titles = ['1 vs. 3', '1 vs. 5', '3 vs. 5', '4 vs. 5']
    for ax, title in zip(axes, titles):
        ax.set_xticks(list(range(1, 20)))
        ax.set_title(title)
        # ax.legend(loc='lower right')
    lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=4)
    fig.text(0.5, 0.04, 'The number of components', ha='center', va='center', fontsize=14)
    fig.text(0.06, 0.5, 'Accuracy', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)  # 调整两幅子图的间距
    plt.show()


if __name__ == '__main__':
    read_and_visual('record.txt')
