import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import cda.ut as ut
import cda.g as g
# 读取外部数据
iris = pd.read_csv("data/iris.txt", header= None)
print(iris.shape)
print(iris.head())

# 调用质心选取函数
iris_cent = ut.dotCenter(iris, 3)
print(iris_cent)

# 调用距离计算（欧氏距离）
q = ut.distEclud(iris.iloc[0, :4].values, iris_cent)
print(q)
print(np.where(q == q.min()))
print(np.where(q == q.min())[0])

# 误差平方和
iris_SSE = g.gKcLearningCurve(iris)
