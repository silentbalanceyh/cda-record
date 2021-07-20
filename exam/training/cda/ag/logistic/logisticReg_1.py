"""
#    SGD求解逻辑回归
"""
import numpy as np
import pandas as pd

import cda.ut as ut
def logisticReg_1(dataSet, eps = 0.01, numIt = 50000):
    # 外置参数, 正规化处理 xMat
    xMat = np.mat(dataSet.iloc[:, :-1].values)
    yMat = np.mat(dataSet.iloc[:, -1].values).T
    xMat = ut.normRegularize(xMat)

    dataSet = pd.DataFrame(xMat).sample(numIt, replace = True)
    dataSet.index = range(dataSet.shape[0])

    m, n = xMat.shape
    weights = np.zeros((n, 1))
    for i in range(m):
        grad = xMat[i].T * (ut.vSigmoid(xMat[i] * weights) - yMat[i])
        weights = weights - eps * grad
    return weights