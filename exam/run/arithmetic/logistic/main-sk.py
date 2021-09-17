"""
逻辑回归算法（基于Scikit-Learn）
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

testSet = pd.read_table('data/testSet.txt', header = None)
print(testSet.head())

clf = LogisticRegression()
clf.fit(testSet.iloc[:, :-1], testSet.iloc[:, -1])

print(clf.coef_)        # 截距
print(clf.intercept_)   # 方程系数

ret = clf.predict(testSet.iloc[:, :-1])

from sklearn.metrics import accuracy_score
##
score = accuracy_score(testSet.iloc[:, -1], ret)
print(ret)
print(score)

"""
逐步回归算法
"""
import cda.ag.logistic as log
aba = pd.read_table('data/abalone.txt', header = None)
returnMat = log.stageWise(aba, 0.001, 5000)
print(returnMat[-1])
ab = pd.DataFrame(returnMat)
for i in range(ab.shape[1]):
    plt.plot(ab.index, ab.iloc[:, i])
plt.show()