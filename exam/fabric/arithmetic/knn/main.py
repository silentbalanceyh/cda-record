"""
KNN算法详细代码实现（自己实现KNN流程）
"""

"""
「1」导入部分

导入必要的包，如果是在Jupyter中，需要执行下边代码以保证绘图

%matplotlib inline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cda.ag.knn as knn
import cda.g as g
"""
「2」函数定义部分
"""
# 函数1：定义创建数据集的函数
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

"""
「3」代码执行主流程
"""
group, labels = createDataSet()
# 显示图片
plt.plot(group[:, 0], group[:, 1], 'ro')
plt.show()

# 1. 执行KNN
# 维度不一样时使用链接
# vstack - 纵向链接
# hstack - 横向链接
train = np.vstack([group, [0, 0]])
labels.append('B')
# 生成DataFrame
train = pd.DataFrame({
    'x1': train[:, 0],
    'x2': train[:, 1],
    'labels': labels
})
train = train.reindex(['x1'] + ['x2'] + ['labels'], axis=1)
# 创建测试集
p1 = [1, 2]
p2 = [0, 1]
test = pd.DataFrame({
    'x1': p1,
    'x2': p2
})
# 测试算法运行
result = knn.classify0_1(train, test, 3)
"""
运行结果
   x1  x2 predict
0   1   0       B
1   2   1       A
"""
print(result)
g.gNearestPoint(train, result)
# # 2. 执行KNN并可视化呈现结果
# result.columns = ['x1', 'x2', 'labels']
# print(result)
#
# # 和train数据集进行拼接
# input = pd.concat([train, result], ignore_index= True)
# print(input)
#
# # 拼接完成后设置指标用于作图
# # - 第一列用于标识字符串标签对应的数值型标签
# # - 第二列用于标识区分测试集和训练集
# input['Ind1'] = 1
# for i in range(input.shape[0]):
#     if 'B' == input.iloc[i, 2]:
#         input.iloc[i, 3] = 0
# input['Ind2'] = [1,1,1,1,1,0.5,0.5]
# plt.scatter(input.iloc[:, 0], input.iloc[:, 1], s = 200 * input.iloc[:, 4], c = input.iloc[:, 3])
# plt.show()