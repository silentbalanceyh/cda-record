import random
# 切分数据集
def randSplit(dataSet, rate):
    l = list(dataSet.index)
    random.shuffle(l)
    n = dataSet.shape[0]
    m = int(n * rate)
    train = dataSet.loc[range(m), :]
    test = dataSet.loc[range(m, n), :]
    dataSet.index = range(dataSet.shape[0])
    test.index = range(test.shape[0])
    return train, test