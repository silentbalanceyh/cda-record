import pandas as pd
import cda.ag.dt as dt
# 原始数据
customer = pd.read_csv('data/customer.csv')
print(customer)
# 处理数据
c45Nodes = dt.c45SplitNode(customer)
print(c45Nodes)