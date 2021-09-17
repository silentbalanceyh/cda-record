import numpy as np
# 多分类模型准确率
import pandas as pd

from cda.ut.toolkit import distEclud
# 计算精准率 - 决策树辅助
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
# 交叉验证 - 决策树辅助
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
# 轮廓系数
def evaSilhouetteCoe(result_set, centroids):
    m, n = result_set.shape
    nc = len(centroids)
    for i in range(nc):
        result_set[n + i] = 0
    result_list = []
    for i in range(nc):
        result_temp = result_set[result_set.iloc[:, n-1] == i]
        result_temp.index = range(result_temp.shape[0])
        result_list.append(result_temp)
    for i in range(m):
        for j in range(nc):
            result_set.iloc[i, n+j] = distEclud(result_set.iloc[i, :n-4].values, result_list[j].iloc[:, :n-4].values).mean()
    result_set['a'] = 0
    result_set['b'] = 0
    for i in range(m):
        l_temp = []
        for j in range(nc):
            if(result_set.iloc[i, n-1] == j):
                result_set.loc[i, 'a'] = result_set.iloc[i, n+j]
            else:
                l_temp.append(result_set.iloc[i, n+j])
        result_set.loc[i, 'b'] = np.array(l_temp).max()
    result_set['s'] = (result_set.loc[:, 'b'] - result_set.loc[:, 'a']) / result_set.loc[:, "a":"b"].max(axis = 1)
    return result_set['s'].mean()
# SSE 计算（残差平方和）
def evaSseCal(dataSet, regress):
    n = dataSet.shape[0]
    y = dataSet.iloc[:, -1].values
    ws = regress(dataSet)
    yhat = dataSet.iloc[:, :-1].values * ws
    yhat = yhat.reshape([n,])
    rss = np.power(yhat - y, 2).sum()
    return rss
# RSS 计算
def evaRssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()
# R-Square计算
def evaRSquare(dataSet, regress):
    sse = evaSseCal(dataSet, regress)
    y = dataSet.iloc[:, -1].values
    sst = np.power(y - y.mean(), 2).sum()
    return 1 - sse / sst
# 置信度计算函数（频繁二项集）
def evaConf2(freqSet, H, supportData, brl, minConf = 0.5):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, '置信度: ', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH
# 置信度计算（频繁多项集）
def evaConfN(freqSet, H, supportData, brl, minConf = 0.7, aprioriGen = None):
    Hmp = True
    while Hmp:
        Hmp = False
        H = evaConf2(freqSet, H, supportData, brl, minConf)
        H = aprioriGen(H)
        Hmp = not (H == [] or len(H[0]) == len(freqSet))