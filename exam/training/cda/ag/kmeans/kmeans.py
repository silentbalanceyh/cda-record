import numpy as np
import pandas as pd

import cda.ut as ut
#       基础版
#       KMeans算法实现流程
"""
特别注意：
1. 设置统一的操作对象 result_set
   为了调用和使用方便，此处将clusterAssment容易转换为DataFrame并与输入的DataFrame合并，组成的对象可作为后续统一调用对象，该对象
   保存了原始数据，也保存了迭代运算中间结果，包含所属簇标记和距质心距离，该对象同时也作为最终函数返回结果
2. 判断质心是否发生改变的条件
   K-Means中判断质心是否改变——判断是否继续下一次迭代的一句：不是某点距离新的质心距离变短，而是某点新的距离向量（到各质心的距离）中
   最短的分量位置是否发生变化，即质心变化后某点是否归属于另外的簇。在质心变化导致各点所属簇发生变化过程中点到质心的距离不一定变短，
   即判断条件下不能用
   if not (result_set.iloc[:, -1] == result_set.iloc[:, -2]).all()
3. 合并DataFrame后索引值为n的列
   小技巧：迅速定位DataFrame合并后列索引，即两个DF合并后后者的第一列在合并后的DF索引值为n，第二列索引值为n+1
4. 质心和类别一一对应
   最后生成的结果中，centroids的行标为result_set中各点所属类别
"""
def kMeans(dataSet, k, distMeas = ut.distEclud, createCent = ut.dotCenter):
    """
    :param dataSet:     输入的数据集
    :param k:           K值（选择的质心的数量）
    :param distMeas:    距离计算函数，默认为欧式距离
    :param createCent:  质心选择函数，默认随机选择
    :return:
    """

    """
    利用质心函数选择质心，执行K-Means时，要不断迭代质心，因此需要两个可迭代容器完成该目标
    1. 第一个容器用于存放和更新质心，该容器考虑使用list，它不仅可迭代，同时不同元素索引位置也可直接标记和区分质心（各簇编号）
    2. 第二个容器则要记录、保存、更新各点到质心的距离，并且方便比较，该容器使用三列的数组执行
        2.1. 第一列存放最近一次计算完成后某点到各质心的最短距离
        2.2. 第二列用于记录最近一次计算完成后根据最短距离得到的代表对应质心的数值索引（所属簇、质心编号）
        2.3. 第三列用于存放更新上一次某点所对应的质心编号（某点所属簇）
    """
    m = dataSet.shape[0]
    n = dataSet.shape[1]
    centroids = createCent(dataSet, k)

    """
    初始化流程
    执行第一次迭代，计算每点到中心点之间的距离，并记录最短距离判定该点所属簇，之前
    1. 初始默认距离设为无穷大
    2. 默认簇类别设置成-1
    """
    clusterAssment = np.zeros((m, 3))
    clusterAssment[:, 0] = np.inf     # 距离无穷大
    clusterAssment[:, 1:3] = -1       # 簇类别

    """
    当某点簇发生变化时，需要更新质心，在欧式距离定义的空间内，质心选取方法是计算该簇中各点的均值
    """
    result_set = pd.concat([dataSet, pd.DataFrame(clusterAssment)], axis=1, ignore_index=True)

    clusterChanged = True
    while clusterChanged:
        # 这一行必须有
        clusterChanged = False
        for i in range(m):
            dist = distMeas(dataSet.iloc[i, :n-1].values, centroids)
            result_set.iloc[i, n] = dist.min()
            result_set.iloc[i, n+1] = np.where(dist == dist.min())[0]

        """
        比较质心是否发生变化，并根据决定是否进一步动迭代，由于预先比较列的值设置位-1，因此无论是否发生下一次迭代，簇的划分类是否
        发生变化都可以使用 all 进行判断
        注意：
        1. 当且仅当所有值相同时，才会返回True，可根据此指定外层控制循环条件
        2. - any函数，用来判断布尔值组成的序列中是否包含True，有一个分量为True则是True
        """
        clusterChanged = not (result_set.iloc[:, -1] == result_set.iloc[:, -2]).all()
        if clusterChanged:
            cent_df = result_set.groupby(n+1).mean()
            centroids = cent_df.iloc[:, :n-1].values
            result_set.iloc[:, -1] = result_set.iloc[:, -2]
    return centroids, result_set