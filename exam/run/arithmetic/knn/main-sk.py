"""
KNN算法详细代码实现（基于Scikit-Learn）
"""
import numpy as np

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

"""
创建数据集
"""
group, labels = createDataSet()

"""
先实例一个方法，然后再在数据集中训练成一个模型
"""

from sklearn.neighbors import KNeighborsClassifier
k = 3
knn = KNeighborsClassifier(n_neighbors = 3)
# 如果没有则抛出错误
# This KNeighborsClassifier instance is not fitted yet.
# Call 'fit' with appropriate arguments before using this estimator.
knn.fit(group, labels)
"""
输入预测数据，利用模型predict方法对未知样本点进行分类
"""

test = np.array([[1, 0.5], [0, 0]])
result = knn.predict(test)
print(result)