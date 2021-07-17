import numpy as np
# 多分类模型准确率
import pandas as pd
# 计算精准率
def evaAccuracy(dataSet):
    m = dataSet.shape[0]
    res = (dataSet.iloc[:, -1] == dataSet.iloc[:, -2]).value_counts()
    acc = res.loc[True] / m
    print("模型精准率 Accuracy = %f" % acc)
    return acc
# 混淆矩阵
def evaConfusionMatrix(dataSet, pos, neg):
    TP = dataSet.loc[(dataSet.iloc[:, -1] == pos) & (dataSet.iloc[:, -2] == pos),].shape[0]
    FN = dataSet.loc[(dataSet.iloc[:, -1] == neg) & (dataSet.iloc[:, -2] == pos),].shape[0]
    TN = dataSet.loc[(dataSet.iloc[:, -1] == neg) & (dataSet.iloc[:, -2] == neg),].shape[0]
    FP = dataSet.loc[(dataSet.iloc[:, -1] == pos) & (dataSet.iloc[:, -2] == neg),].shape[0]
    dataSet_ac = (TP + TN) / (TP + TN + FP + FN)
    dataSet_pr = TP / (TP + FP)
    dataSet_re = TP / (TP + FN)
    dataSet_sp = TN / (TN + FP)
    dataSet_F = 2 * dataSet_pr * dataSet_re / (dataSet_pr + dataSet_re)
    print("准确率：%f" % dataSet_ac)
    print("精确度：%f" % dataSet_pr)
    print("召回率：%f" % dataSet_re)
    print("特意度：%f" % dataSet_sp)
    print("F指标：%f" % dataSet_F)
    return [dataSet_ac,dataSet_pr,dataSet_re,dataSet_sp,dataSet_F]
# 交叉验证
def evaCrossVali(dataSet, randSplit, classify, n, k):
    """
    :param dataSet:     进行分类测试的数据集
    :param randSplit:   自定义的数据集随机切分函数
    :param classify:    自定义的KNN分类器
    :param n:           数据集等分的个数
    :param k:           KNN中的最近邻元素个数
    :return:
    """
    sp = randSplit(dataSet, n)
    result = np.array([])
    for i in range(n):
        test = sp[0]
        del sp[0]
        train = pd.concat(sp)
        train.index = range(train.shape[0])
        test.index = range(test.shape[0])
        test = classify(train, test, k)
        result = np.append(result, evaAccuracy(test))
        test = pd.DataFrame(test.drop(['predict'], axis = 1))
        sp.append(test)
    return result, result.mean(), result.var()

