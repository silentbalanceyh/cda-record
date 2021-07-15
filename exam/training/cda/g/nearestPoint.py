# 将绘图处理成函数
import pandas as pd
import matplotlib.pyplot as plt
# 绘制散点图工具函数
def nearestPoint(train, result):
    global output
    m = train.shape[0]
    n = result.shape[0]
    uq = train.iloc[:,-1].unique()
    k = uq.shape[0]
    ind1 = pd.Series(index = range(m + n))
    ind2 = pd.Series(index = range(m + n))
    ind2[:m] = 1
    ind2[m:] = 2
    if(train.shape[1] == result.shape[1]):
        result.columns = train.columns
        input = pd.concat([train, result], ignore_index= True)
        for i in range(k):
            ind1[input.loc[:, 'labels'] == uq[i]] = i + 1
        output = pd.concat([input, ind1, ind2], axis = 1)
    plt.scatter(output.iloc[:, 0], output.iloc[:, 1], s = 200 * output.iloc[:, -1], c = output.iloc[:, -2])
    plt.show()      # IDE 中呈现必备
    return output