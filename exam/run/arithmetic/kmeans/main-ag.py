import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cda.ag.kmeans as kmeans
import cda.g as g
import cda.ut as ut
# 算法验证
testSet = pd.read_table("data/testSet.txt", header= None)
print(testSet.head())
print(testSet.shape)

# 二维平面观察分布
plt.plot(testSet.iloc[:, 0], testSet.iloc[:, 1], 'o')
plt.show()

ze = pd.DataFrame(np.zeros(80).reshape(-1, 1))
test_set = pd.concat([testSet, ze], axis= 1, ignore_index=True)
print(test_set.head())

# 算法调用验证
test_cent, test_cluster = kmeans.kMeans(test_set, 4)
print(test_cent)
print(test_cluster.head())

# 查看最终结果
plt.scatter(test_cluster.iloc[:,0], test_cluster.iloc[:,1], c = test_cluster.iloc[:,-1])
plt.plot(test_cent[:, 0], test_cent[:, 1], 'o', color = 'red')
plt.show()

# 在已有数据集上进行测试
test_SSE = g.gKcLearningCurve(testSet)

# 第二种算法
test_cent, test_cluster = kmeans.biKmeans(test_set, 4)
# 查看最终结果
plt.scatter(test_cluster.iloc[:,0], test_cluster.iloc[:,1], c = test_cluster.iloc[:,-1])
plt.plot(test_cent[:, 0], test_cent[:, 1], 'o', color = 'red')
plt.show()

# 轮廓系数
sil = []
for i in range(1, 7):
    centroids, result_set = kmeans.biKmeans(test_set, i+1)
    sil.append(ut.evaSilhouetteCoe(result_set, centroids))
plt.plot(range(2, 8), sil, '--o')
plt.show()