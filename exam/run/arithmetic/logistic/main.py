"""
逻辑回归算法自己实现
"""
import pandas as pd
import numpy as np
import cda.ag.logistic as log
import cda.ut as ut
testSet = pd.read_table('data/testSet.txt', header = None)
print(testSet.head())
# 直接执行算法
ws = log.logisticReg_0(testSet, eps = 0.01, numIt = 500)
print(ws)
# 中间步骤
xMat = np.mat(testSet.iloc[:, :-1].values)
yMat = np.mat(testSet.iloc[:, -1].values).T
xMat = ut.normRegularize(xMat)
fRet = (xMat * ws).A.flatten()
print(fRet)
p = ut.vSigmoid(xMat * ws).A.flatten()
print(p)
## 迭代计算
for i, j in enumerate(p):
    if j < 0.5:
        p[i] = 0
    else:
        p[i] = 1
## 转换后
print(p)

train_error = (np.fabs(yMat.A.flatten() - p)).sum()
print(train_error)
train_error_rate = train_error / yMat.shape[0]
print(train_error_rate)
## BGD 结果函数（追加 xMat 和 yMat）
bret = log.logisticAcc(testSet, log.logisticReg_0)
print(bret)
## SGD 结果函数
newTSet = pd.read_table('data/testSet.txt', header = None)
sret = log.logisticAcc(newTSet, log.logisticReg_1)
print(sret)