
import numpy as np
import cda.ut as ut

def logisticAcc(dataSet, method, eps = 0.01, numIt = 500):
    weights = method(dataSet, eps, numIt)
    # 外置参数, 正规化处理 xMat
    xMat = np.mat(dataSet.iloc[:, :-1].values)
    yMat = np.mat(dataSet.iloc[:, -1].values).T
    xMat = ut.normRegularize(xMat)
    # 执行 sigmod
    p = ut.vSigmoid(xMat * weights).A.flatten()
    for i, j in enumerate(p):
        if j < 0.5:
            p[i] = 0
        else:
            p[i] = 1
    train_error = (np.fabs(yMat.A.flatten() - p)).sum()
    trainAcc = 1 - train_error / yMat.shape[0]
    return trainAcc