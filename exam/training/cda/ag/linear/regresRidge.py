#       岭回归
#       手动构造线性回归
import numpy as np

def regresRidge(dataSet, lam = 0.2):
    xMat = np.mat(dataSet.iloc[:, :-1].values)
    yMat = np.mat(dataSet.iloc[:, -1].values).T
    xTx = xMat.T * xMat
    denom = xTx + np.eye(xMat.shape[1]) * lam
    ws = denom.I * (xMat.T * yMat)
    return ws