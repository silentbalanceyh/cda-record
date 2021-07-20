import random
# ========================== 划分数据集
# 切分数据集 - 决策树辅助
def splitRate(dataSet, rate):
    l = list(dataSet.index)
    random.shuffle(l)
    n = dataSet.shape[0]
    m = int(n * rate)
    train = dataSet.loc[range(m), :]
    test = dataSet.loc[range(m, n), :]
    dataSet.index = range(dataSet.shape[0])
    test.index = range(test.shape[0])
    return train, test

# 交叉验证专用数据集切分 - 决策树辅助
def splitN(dataSet, n):
    l = list(dataSet.index)
    random.shuffle(l)
    dataSet.index = l
    m = dataSet.shape[0]
    splitSet = []
    k = m / n
    for i in range(n):
        if i < (n - 1):
            splitSet.append(dataSet.loc[range(i * int(k), (i+1)*int(k)),:])
        else:
            splitSet.append(dataSet.loc[range(i * int(k), m), :])
    dataSet.index = range(dataSet.shape[0])
    return splitSet
