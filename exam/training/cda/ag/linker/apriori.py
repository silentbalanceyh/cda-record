"""
#    Apriori 算法
"""
import cda.ut as ut
"""
创建函数用于满足最小支持度（查找项集）
"""
def aprioriScanD(D, Ck, minSupport):
    """
    :param D:
    :param Ck:          数据集CK（包含候选集合的列表）
    :param minSupport:  项集最小支持度
    :return:
    """
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                ssCnt[can] = ssCnt.get(can, 0) + 1
    numItems = float(len(D))
    retList = []
    supportData = {}        # 注意初始化类型
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData
"""
生成频繁项集专用函数
"""
def aprioriGen(Lk):
    retList = []
    lenLk = len(Lk)
    k = len(Lk[0])
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-1]; L2 = list(Lk[j])[:k-1]
            L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList
"""
核心算法
"""
def aprioriRun(dataSet, minSupport = 0.5):
    C1 = ut.relC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = aprioriScanD(D, C1, minSupport)
    L = [L1]
    while (len(L[-1]) > 1):
        Ck = aprioriGen(L[-1])
        Lk, supK = aprioriScanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
    return L, supportData
"""
规则挖掘
"""
def aprioriRules(L, supportData, minConf = 0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                ut.evaConfN(freqSet, H1, supportData, bigRuleList, minConf, aprioriGen)
            else:
                ut.evaConf2(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList