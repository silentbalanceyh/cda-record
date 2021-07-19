import random
import numpy as np
# 切分数据集
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

# 交叉验证专用数据集切分
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

# 自动生成随机质心函数
def dotCenter(dataSet, k):
    n = dataSet.shape[1]
    data_min = dataSet.iloc[:, :n-1].min()
    data_max = dataSet.iloc[:, :n-1].max()
    data_mid = (data_max + data_min) / 2
    data_ran = (data_max - data_min) / 2
    data_cent_o = np.random.random((k, n-1))
    data_cent = data_cent_o * list(data_ran) + list(data_mid)
    return data_cent

# 欧式距离
def distEclud(arrayA, arrayB):
    dist_o = arrayA - arrayB
    return np.sum(np.power(dist_o, 2), axis=1)

# sigmoid 函数
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))
