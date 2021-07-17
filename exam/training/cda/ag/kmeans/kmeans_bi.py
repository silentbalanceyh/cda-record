import numpy as np
import pandas as pd

import cda.ut as ut
from cda.ag.kmeans.kmeans import kMeans
#       使用二分K均值算法选择质心
#       KMeans算法实现流程（更优化的算法）
def biKmeans(dataSet, k, distMeans = ut.distEclud):
    m = dataSet.shape[0]
    n = dataSet.shape[1]
    centroids, result_set = kMeans(dataSet, 2)
    j = 2
    while j < k:
        result_tmp = result_set.groupby(n+1).sum()
        clusterAssment = pd.concat([pd.DataFrame(centroids), result_tmp.iloc[:, n]], axis = 1, ignore_index=True)
        lowestSSE = clusterAssment.iloc[:, n-1].sum()
        centList = []
        sseTotle = np.array([])
        for i in clusterAssment.index:
            df_temp = result_set.iloc[:, :n][result_set.iloc[:, -1] == i]
            df_temp.index = range(df_temp.shape[0])
            cent, res = kMeans(df_temp, 2, distMeans)
            centList.append(cent)
            sseSplit = res.iloc[:, n].sum()
            sseNotSplit = result_set.iloc[:, n][result_set.iloc[:, -1] != i].sum()
            sseTotle = np.append(sseTotle, sseSplit + sseNotSplit)
        min_index = np.where(sseTotle == sseTotle.min())[0][0]
        clusterAssment = clusterAssment.drop([min_index])
        centroids = np.vstack([clusterAssment.iloc[:, :n-1].values, centList[min_index]])
        result_set = kMeans_assment(dataSet, centroids)
        j = j+1
    return centroids, result_set

def kMeans_assment(dataSet, centroids, distMeans = ut.distEclud):
    m = dataSet.shape[0]
    n = dataSet.shape[1]
    clusterAssment = np.zeros((m, 3))
    clusterAssment[:, 0] = np.inf
    clusterAssment[:, 1:3] = -1
    result_set = pd.concat([dataSet, pd.DataFrame(clusterAssment)], axis=1, ignore_index=True)
    for i in range(m):
        dist = distMeans(dataSet.iloc[i, :n-1].values, centroids)
        result_set.iloc[i, n] = dist.min()
        result_set.iloc[i, n+1] = np.where(dist == dist.min())[0]
        result_set.iloc[:, -1] = result_set.iloc[:, -2]
    return result_set