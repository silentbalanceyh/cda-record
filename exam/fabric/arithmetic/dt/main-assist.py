import numpy as np
import pandas as pd

def createDataSet():
    group = np.array([
        [1,2],
        [1,0],
        [2,1],
        [0,1],
        [0,0]
    ])
    labels = np.array([0,1,0,1,0])
    dataSet = pd.concat([pd.DataFrame(group), pd.DataFrame(labels)], axis = 1, ignore_index=True)
    return dataSet

dataSet = createDataSet()
import cda.ut as ut
# 错误率
ret1 = ut.dtClError(dataSet)
print(ret1)
# 熵
ret2 = ut.dtEntropy(dataSet)
print(ret2)
# Gini
ret3 = ut.dtGini(dataSet)
print(ret3)