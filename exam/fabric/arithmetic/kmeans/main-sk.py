"""
KMeans算法详细代码实现（基于Scikit-Learn）
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# 算法验证
testSet = pd.read_table("data/testSet.txt", header= None)
print(testSet.head())
print(testSet.shape)

# 数据分布
kmean_set = testSet.values
plt.plot(kmean_set[:, 0], kmean_set[:, 1], 'o')
plt.show()

# SK 算法
n_cluster = 4
kmeans = KMeans(n_cluster)
kmeans.fit(kmean_set)

# 输出结果
test_cluster = kmeans.predict(kmean_set)
test_cent = kmeans.cluster_centers_
print(test_cent)
# 查看结果
plt.scatter(kmean_set[:, 0], kmean_set[:, 1], c = test_cluster)
plt.plot(test_cent[:, 0], test_cent[:, 1], 'o', color = 'red')
plt.show()