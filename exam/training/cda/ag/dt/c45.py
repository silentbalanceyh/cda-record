
"""
#       基础版
#       C4.5算法实现流程
"""
"""
C4.5分裂的树算法
"""
import cda.ut as ut
import pandas as pd
import numpy as np

def c45SplitNode(dataSet):
    m,n = dataSet.shape
    ent_fn = ut.dtEntropy(dataSet)
    gr = np.array([])
    for i in pd.DataFrame(dataSet.iloc[:, :n-1]).columns:
        levels = dataSet.loc[:, i].value_counts(sort = False).index
        data_temp = []
        ent_temp = []
        iv_temp = np.array([])
        for j in levels:
            data_temp.append(dataSet[dataSet.loc[:, i] == j])
        for k in data_temp:
            ent_temp.append(ut.dtEntropy(k) * k.shape[0] / m)
            iv_temp = np.append(iv_temp, k.shape[0] / m)
        gn = ent_fn - sum(ent_temp)
        iv = (-iv_temp * np.log2(iv_temp)).sum()
        gr = np.append(gr, gn / iv)
    col_index = np.where(gr == gr.max())[0][0]
    levels = dataSet.iloc[:, col_index].value_counts(sort = False).index
    data_temp = []
    for j in levels:
        data_temp.append(dataSet[dataSet.iloc[:, col_index] == j])
    for z in data_temp:
        del z[z.columns[col_index]]
    return gr, data_temp

def c45TreeGrowth(dataSet):
    m, n = dataSet.shape
    treeNode = [dataSet]
    der = True
    while der:
        len_b = len(treeNode)
        for i in range(len_b):
            if treeNode[i].shape[1] > 1:
                gr, data_temp = c45SplitNode(treeNode[i])
                if gr.min() > 0:
                    del(treeNode[i])
                    treeNode.extend(data_temp)
        len_a = len(treeNode)
        der = not (len_a == len_b)
    return treeNode
