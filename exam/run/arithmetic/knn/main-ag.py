import pandas as pd

import cda.ut as ut
import cda.ag.knn as knn

pd.set_option('display.width', 200)            # 打印宽度
pd.set_option('display.max_columns', None)      # 打印最大列
pd.set_option('display.max_rows', None)         # 打印最大行

iris = pd.read_csv('data/iris.txt', header = None)
iris.columns = [
    'sepal_length_cm',
    'sepal_width_cm',
    'petal_length_cm',
    'petal_width_cm',
    'class'
]
# 手动切分数据集
train, test = ut.splitRate(iris, 0.8)
print(train.head())
print(test.head())
# 训练集
trained = knn.classify0_1(train, test, 3)
print(trained)