
import numpy as np
#       基础版
#       手动构造线性回归
def regresStand(dataSet):
    """
    :param dataSet:     输入数据集
    :return:
    """
    """
    矩阵创建，在 NumPy 中使用 mat 创建矩阵（该函数需要二维数组）
    """
    xMat = np.mat(dataSet.iloc[:,:-1].values)
    yMat = np.mat(dataSet.iloc[:,-1].values).T
    """
    矩阵运算
    NumPy中包含了一个 linalg 库给矩阵提供相关运算方法
    1. 矩阵转秩：m.T
    2. 剧本乘法：m * m, a * a
    3. 矩阵行列式：np.linalg.det(m)
    4. 求逆矩阵：m.I
    """
    xTx = xMat.T * xMat
    if 0 == np.linalg.det(xTx):
        print("该矩阵是 Singular的，没有可逆矩阵")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws