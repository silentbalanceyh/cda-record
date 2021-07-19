"""
线性回归算法自己实现
"""

import pandas as pd
import matplotlib.pyplot as plt

import cda.ag.linear as lg
import cda.ut as ut
ex0 = pd.read_table('data/ex0.txt', header = None)
print(ex0.head())
# 执行线性回归
ws = lg.regresStand(ex0)
print(ws)
# 图示结果
yHat = ex0.iloc[:, :-1].values * ws
plt.plot(ex0.iloc[:, 1], ex0.iloc[:, 2], 'o')
plt.plot(ex0.iloc[:, 1], yHat)
plt.show()
# 残差平方和
rss = ut.evaSseCal(ex0, lg.regresStand)
print(rss)
rSquare = ut.evaRSquare(ex0, lg.regresStand)
print(rSquare)