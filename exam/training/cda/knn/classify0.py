import numpy as np
#       基础版
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
