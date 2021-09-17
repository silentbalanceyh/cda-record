
import cda.ut as ut

import cda.ag.linker as lnk

def loadDataSet():
    return [
        [1,3,4],
        [2,3,5],
        [1,2,3,5],
        [2,5]
    ]
# 项集
dataSet = loadDataSet()
C1 = ut.relC1(dataSet)
print(C1)
# 基础算法
D = list(map(set, dataSet))
L1, scanData = lnk.aprioriScanD(D, C1, minSupport= 0.5)
print("--- Apriori算法细节 ---")
print(L1)
print(scanData)
# 频繁项集
gen = lnk.aprioriGen(L1)
print(gen)

# 算法执行流程
print("--- Apriori算法执行 ---")
L, supportData = lnk.aprioriRun(dataSet, minSupport= 0.5)
print(L)
# 计算置信度，置信度流程
print("---（二项集置信度） ---")
freqSet = L[1][1]
H1 = [frozenset([item]) for item in freqSet]
print("步骤1: freqSet = ", freqSet)
print("步骤2: H1 = ", H1)
brl = []
retConf = ut.evaConf2(freqSet, H1, supportData, brl, minConf= 0.5)
print("步骤3：结果 = ",retConf, " brl = ", brl)

print("---（多项集置信度） ---")
freqSet1 = L[2][0]
H2 = [frozenset([item]) for item in freqSet1]
H3 = lnk.aprioriGen(H2)
print("步骤1: freqSet = ", freqSet1)
print("步骤2: H1 = ", H3)
brl1 = []
retConf1 = ut.evaConfN(freqSet1, H3, supportData, brl1, minConf= 0.5)
print("步骤3：结果 = ",retConf1, " brl = ", brl)
