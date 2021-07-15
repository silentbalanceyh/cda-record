# 将绘图处理成函数
import pandas as pd
import matplotlib.pyplot as plt

import cda.pre as pre
# 绘制散点图工具函数
def gNearestPoint(train, result):
    global output
    m = train.shape[0]
    n = result.shape[0]
    uq = train.iloc[:,-1].unique()
    k = uq.shape[0]
    ind1 = pd.Series(index = range(m + n))
    ind2 = pd.Series(index = range(m + n))
    ind2[:m] = 1
    ind2[m:] = 2
    if(train.shape[1] == result.shape[1]):
        result.columns = train.columns
        input = pd.concat([train, result], ignore_index= True)
        for i in range(k):
            ind1[input.loc[:, 'labels'] == uq[i]] = i + 1
        output = pd.concat([input, ind1, ind2], axis = 1)
    plt.scatter(output.iloc[:, 0], output.iloc[:, 1], s = 200 * output.iloc[:, -1], c = output.iloc[:, -2])
    plt.show()      # IDE 中呈现必备
    return output

# 绘制K值学习曲线
def gKLearningCurve(classify, train, test, k):
    """
    :param classify: 用于指定分类函数，此处可以选择之前定义的 classify0_1 和在此基础上优化过的 classify0_2
    :param train:    训练集
    :param test:     测试集
    :param k:        用于指定K值选取范围，这里需要注意，选取范围是1到k，左右均包含精确率
    :return:
    """
    yAc = []
    for i in range(k):
        yAc.append(pre.evaAccuracy(classify(train, test, i + 1)))
    plt.plot(range(1, k+1), yAc, '-o', color='black')
    return yAc

# 交叉验证修正的K值学习曲线
def gKLearningCurve_1(dataSet, classify, n, k):
    """

    :param dataSet:     输入数据集
    :param classify:    用于指定分类函数，此处可以选择之前定义的 classify0_1 和在此基础上优化过的 classify0_2
    :param n:           交叉验证中数据集随机切分的数量
    :param k:           用于指定K值选取范围，这里需要注意，我们选取范围是从1到k
    :return:
    """
    yAc_mean = []
    yAc_up = []
    yAc_down = []
    for i in range(k):
        result_cv, result_mean, result_var = pre.evaCrossVali(dataSet, pre.splitN, classify, n, i + 1)
        yAc_mean.append(result_mean)
        yAc_up.append(result_mean + result_var)
        yAc_down.append(result_mean - result_var)
        plt.plot(range(1, k+1), yAc_mean, '-o', color='black')
        plt.plot(range(1, k+1), yAc_up, '-o', color='red')
        plt.plot(range(1, k+1), yAc_down, '-o', color='blue')
        return yAc_mean, yAc_up, yAc_down