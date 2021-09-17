import pandas as pd
import matplotlib.pyplot as plt

import cda.ag.linear as lg
import cda.ut as ut
import cda.g as g
aba = pd.read_table('data/abalone.txt', header = None)
print(aba.head())
# 最后一列为表前列，由于第一列是分类变量，所以将第一列的值全部设置为1
aba.iloc[:, 0] = 1
print(aba.head())
# 带入模型运算
rws = lg.regresRidge(aba)
ws = lg.regresStand(aba)
print(rws)
print(ws)
rsRws = ut.evaRSquare(aba, lg.regresRidge)
rsWs = ut.evaRSquare(aba, lg.regresStand)
print(rsRws)
print(rsWs)

# 岭回归图
abaO = pd.read_table('data/abalone.txt', header = None)
rightWt = g.gRidge(abaO)
plt.plot(rightWt)
plt.xlabel('log(lambda)')
plt.ylabel('weights')
plt.show()
