import numpy as np
# 0-1 标准化
def normMinMax(dataSet):
    maxDf = dataSet.max()
    minDf = dataSet.min()
    normSet = (dataSet - minDf) / (maxDf - minDf)
    return normSet

# Z-Score 标准化
def normZScore(dataSet):
    stdDf = dataSet.std()
    meanDf = dataSet.mean()
    normSet = (dataSet - meanDf) / stdDf
    return normSet

# Sigmod 压缩法
def normSigmod(dataSet):
    normSet = 1 / (1 + np.exp(-dataSet))
    return normSet

# 利用 DataFrame 数据格式和矩阵计算梯度下降
def normRegularize(xMat):
    inMat = xMat.copy()
    inMeans = np.mean(inMat, axis = 0)
    inVar = np.std(inMat, axis = 0)
    inMat = (inMat - inMeans) / inVar
    return inMat