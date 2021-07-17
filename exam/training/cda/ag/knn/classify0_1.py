
import pandas as pd

#       优化版
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
