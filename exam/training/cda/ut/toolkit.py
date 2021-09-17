
import numpy as np
# ========================== 点函数
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

# ========================== 距离函数
# 欧式距离
def distEclud(arrayA, arrayB):
    dist_o = arrayA - arrayB
    return np.sum(np.power(dist_o, 2), axis=1)
# 曼哈顿距离（略）

# ========================== 基础函数
# vSigmoid 函数
def vSigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))
# vSquare 函数
def vSquare(x):
    return x ** 2
# ========================== 决策树辅助函数
# 错误率：Classification Error
def dtClError(dataSet):
    m = dataSet.shape[0]
    iMax = dataSet.iloc[:, -1].value_counts(ascending= False)[0]
    return 1 - iMax / m
# 熵函数：Entropy
def dtEntropy(dataSet):
    m = dataSet.shape[0]
    iSet = dataSet.iloc[:, -1].value_counts()
    p = iSet / m
    return (-p * np.log2(p)).sum()
# Gini指数：Gini
def dtGini(dataSet):
    m = dataSet.shape[0]
    iSet = dataSet.iloc[:, -1].value_counts()
    p = iSet / m
    return 1 - (np.power(p, 2)).sum()

# ========================== 关联规则
# 创建利用事务集一项集
def relC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))