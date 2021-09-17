"""
#    SGD随机梯度下降法
"""
import numpy as np

import cda.ut as ut

def gradDescent_1(dataSet, eps = 0.01, numIt = 5000):
    """
    :param dataSet:  梯度下降计算的数据集
    :param eps:      步长，默认值 0.01
    :param numIt:    迭代次数，50000次默认
    :return:
    """
    dataSet = dataSet.simple(numIt, replace = True)
    dataSet.index = range(dataSet.shape[0])
    xMat = np.mat(dataSet.iloc[:, :-1].values)
    yMat = np.mat(dataSet.iloc[:, -1].values).T
    xMat = ut.normRegularize(xMat)
    yMat = ut.normRegularize(yMat)
    m, n = xMat.shape
    weights = np.zeros((n, 1))
    for k in range(numIt):
        grad = xMat.T * (xMat * weights - yMat) / m
        weights = weights - eps * grad
    return weights