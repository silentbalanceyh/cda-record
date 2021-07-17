import pandas as pd

import cda.ut as ut

dfTest = pd.DataFrame({'X1': [1,2,3], 'X2': [2,7,8]})
print(dfTest)
# 0-1 标准化
print("============= 0 - 1 标准化")
print(dfTest.min())
print(dfTest.max())
print(ut.normMinMax(dfTest))

# ZScore 标准化
print("============= ZScore 标准化")
print(dfTest.var())
print(dfTest.std())
print(dfTest.mean())
print(ut.normZScore(dfTest))