"""
#    BGD求解逻辑回归
"""
import numpy as np

import cda.ut as ut
def logisticReg_0(dataSet, eps = 0.01, numIt = 5000):
    xMat = np.mat(dataSet.iloc[:, :-1].values)
    yMat = np.mat(dataSet.iloc[:, -1].values).T
    xMat = ut.normRegularize(xMat)
    m, n = xMat.shape
    weights = np.zeros((n, 1))
    for k in range(numIt):
        grad = xMat.T * (ut.vSigmoid(xMat * weights) - yMat) / m
        weights = weights - eps * grad
    return weights