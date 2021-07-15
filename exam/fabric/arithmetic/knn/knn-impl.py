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

"""
「2」函数定义部分
"""


# 函数1：定义创建数据集的函数
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 函数2：基础版
#       手动构造KNN分类器，分类器1
#       调用如：classify0([0,0], group, labels, 3)
def classify0(inX, dataSet, labels, k):
    """
    :param inX:         为预测的数据集，该数据集需要进行类别划分
    :param dataSet:     训练数据集，是个二维数组，组成内部每个一位数组都是不包含标签的记录
    :param labels:      训练数据集的标签
    :param k:           选取最近邻作为判别标准的个数
    :return:
    """
    '''
    Modify：全局定义 sortedClassCount，防止警告发生（否则会有相关警告）
    '''
    global sortedClassCount

    '''
    使用shape方法返回数组维度组成的元组，该元组的第一个索引为高纬度数组组成第一维度的数组个数，当作用于二维标识数据集的数组时返回结构
    可理解为记录数
    '''
    dataSetSize = dataSet.shape[0]

    '''
    利用np.tile方法将输入测试数据张成一个行数和训练集行数相同、列数为1 * 测试数据列数的二维数组（矩阵），用于和输入数据计算距离使用，
    其中tile函数的用法参考
    
    fabric/segment/numpy/np.tile.py
    
    最后张成的 4 * 2 的数组就能够和输入数组集进行减法运算了，当然，由于np.array的广播功能，也可以直接运算
    '''
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

    '''
    利用广播功能，计算每个分量的平方
    '''
    sqDiffMat = diffMat ** 2

    '''
    再计算每个分量的和，即训练样本到每个点各分量的平方和，其中axis标识二维数组按每个内嵌数组求和，或剧本按行求和，返回结果为一个
    一维数组
    '''
    sqDisttances = sqDiffMat.sum(axis=1)

    '''
    利用广播对数组中的每个元素进行开平方运算，所得结果即为intX和各训练数据集中其他点的距离
    '''
    distances = sqDisttances ** 0.5

    '''
    对距离计算结果进行排序，返回结果为数组升序排序后的各分量索引值，即原数组秩排序的结果，此处计算距离的远近排名，实际是为了后续
    选取K个最近邻的点。若要直接对原数组进行重排序，则可直接使用sort方法
    '''
    sortedDistIndicies = distances.argsort()

    '''
    定义空字典，这是在使用 python 基础数据格式进行算法设计时对元素个数进行计数最常用的手段
    
    使用字典进行计数时，把需要计数的对象设置成Key，计数对象累积计数值为Value，配合循环进行计数
    '''
    classCount = {}

    '''
    循环计数
    由于只选取最临近的k个元素，因此只需要提取排序后的前k个分量（排序越靠前，则和预测数值越接近），首先将循环范围设置成0到2，使用
    range函数设置循环范围（测试数据集中是3）
    '''
    for i in range(k):
        '''
        当i取值遍历0到2中的每个元素时，选取labels标签列经秩排序结果对应的第i个元素，即距离由近到远的最近的第i个记录对应的标签
        '''
        voteIlabel = labels[sortedDistIndicies[i]]

        '''
        标签被选取一次，即增加一个计数值
        '''
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

        '''
        最后，对保存了对应元素出现次数的字典进行排序，排序之前需要使用items()方法将字典转换为可遍历的由key-value元组组成的数组
        '''
        sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)

    '''
    最终返回字典根据key排序结果的第一个分量，即最近邻元素所属类别的最多数类别
    此处注意，dict_items对象本身不能进行索引，要利用可便利性转换为list，此时的list为原字典key-value组成的元组所组成的list
    '''
    return sortedClassCount[0][0]


# 函数3：优化版
#       使用高级数据结构优化算法
#       引入 NumPy 和 Pandas，更快的执行效率、简化的IO、设计过程中更简洁
#       可加快分析流程
def classify0_1(train, test, k):
    """
    :param train:   训练集
    :param test:    测试集
    :param k:       选取最近邻作为判别标准的个数
    :return:
    """
    n = train.shape[1] - 1
    m = test.shape[0]
    result = []
    """
    利用 DataFrame 字典属性执行数据导入，结合导入结果可进一步调整数据集列的顺序，符合数据分析习惯
    思考：
    1. 为何此处利用字典生成DF后会"乱序"
    2. 为何不用数组拼接方式生成DF
    """
    for i in range(m):
        """
        利用NumPy数据格式特性优化算法执行流程，首先计算各点之间距离，此处可使用NumPy的广播特性计算，
        Pandas由于架设在NumPy之上，所以也支持广播
        """
        dist = list(((train.iloc[:, :n] - test.iloc[i, :n]) ** 2).sum(1))

        """
        利用DF构建dist和labels之间关系，此处不再使用字典进行关系构造，而直接构造一个临时的DF保存距离和对应标签
        """
        dist_l = pd.DataFrame({'dist': dist, 'labels': (train.iloc[:, n])})

        """
        然后寻找距离最近的k个训练集样本点
        - by：排序的字段
        - ascending: 默认为TRUE（升序排列），若要降序则设置为FALSE
        """
        dr = dist_l.sort_values(by='dist')[: k]

        """
        对DR对象进行labels出现次数的计数，求出结果
        """
        re = dr.loc[:, 'labels'].value_counts()

        """
        re为最终保留的计数结果Series，最后选择计数排名第一的结果为最终预测类别
        """
        result.append(re.index[0])
    result = pd.Series(result)
    test['predict'] = result
    return test


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
result = classify0_1(train, test, 3)
"""
运行结果
   x1  x2 predict
0   1   0       B
1   2   1       A
"""
print(result)

# 2. 执行KNN并可视化呈现结果
result.columns = ['x1', 'x2', 'labels']
print(result)

# 和train数据集进行拼接
input = pd.concat([train, result], ignore_index= True)
print(input)

# 拼接完成后设置指标用于作图
# - 第一列用于标识字符串标签对应的数值型标签
# - 第二列用于标识区分测试集和训练集
input['Ind1'] = 1
for i in range(input.shape[0]):
    if 'B' == input.iloc[i, 2]:
        input.iloc[i, 3] = 0
input['Ind2'] = [1,1,1,1,1,0.5,0.5]
plt.scatter(input.iloc[:, 0], input.iloc[:, 1], s = 200 * input.iloc[:, 4], c = input.iloc[:, 3])
plt.show()