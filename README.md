
# 实验内容
1. 实现SVM（SVR＋SVC）
   - 用SMO实现SVC，并在数据集Iris和Image Segmentation上进行实验和正则化参数选择。（分类问题 -> SVM对偶 -> SMO）
   - 可借鉴现成SVM软件包，实现SVR对1维y=2sinx+sin2x+e带噪函数的回归。其中e为随机噪声，
     服从N(0, σ^2)(σ=0.1,0.5,1) 和不敏感参数ε=0.01,0.05,1及当同时优化ε后的结果。
2. 分别实现KPCA与PCA在Iris和Image Segmentation数据集上的实验分析。
3. 用KLDA代替KPCA，重复2.

# 需提交
1. 相关的代码及10重交叉验证的描述。
2. 实验分析报告（报告的实验结果分析越丰富，论证越充分，评分将会越高！）。

# 说明
1. 训练集大小：验证集大小：测试集大小=4:3:3/类, 随机重复10轮，结果取平均！ 
2. 对Iris，共3类，请用2类SVC分别以
   - 类-对方式实现（类1VS. 类2；）和（类1VS. 类3）； 
   - 类1&2 VS. 类3. 
3. 对Image Segmentation，用类似方式实现（上交时请注明）。

# 实验环境
1. Mac（M1芯片）；
2. Python编程语言；
3. Pycharm IDE；
4. Python库列表在requirements.txt文件中。
