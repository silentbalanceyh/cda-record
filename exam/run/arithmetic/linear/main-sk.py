"""
线性回归算法（基于Scikit-Learn）
"""
"""
标准线性回归
"""
import pandas as pd
from sklearn import linear_model

ex0 = pd.read_table('data/ex0.txt', header = None)
print(ex0.head())

# 线性回归模型
reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(ex0.iloc[:, :-1].values, ex0.iloc[:, -1].values)
print(reg.coef_)        # 截距
print(reg.intercept_)   # 方程系数

# 结果打印
from sklearn.metrics import mean_squared_error, r2_score
y = ex0.iloc[:, -1].values
yHat = reg.predict(ex0.iloc[:, :-1])
retM = mean_squared_error(y, yHat)
retR = r2_score(y, yHat)
print(retM)
print(retR)

"""
岭回归
"""
aba = pd.read_table('data/abalone.txt', header = None)
print(aba.head())
# 最后一列为表前列，由于第一列是分类变量，所以将第一列的值全部设置为1
aba.iloc[:, 0] = 1
print(aba.head())

ridge = linear_model.Ridge(alpha=.2)
ridge.fit(aba.iloc[:, :-1], aba.iloc[:, -1])
print(ridge.coef_)
print(ridge.intercept_)

"""
Lasso
"""
las = linear_model.Lasso(alpha= 0.01)
las.fit(aba.iloc[:, :-1], aba.iloc[:, -1])
print(las.coef_)
print(las.intercept_)
