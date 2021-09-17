"""
逐步回归算法
"""
import numpy as np
import cda.ut as ut
def stageWise(dataSet, eps = 0.01, numIt = 100):
    xMat = np.mat(dataSet.iloc[:, :-1].values)
    yMat = np.mat(dataSet.iloc[:, -1].values).T
    yMean = np.mean(yMat, axis = 0)
    yMat = yMat - yMean
    xMat = ut.normRegularize(xMat)

    m,n = xMat.shape
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = ut.evaRssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat