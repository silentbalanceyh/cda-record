gr = open('data/groceries.txt')
gro = gr.readlines()
print(gro)

gro[1].strip('\n').split(' ')
l = []
for line in gro:
    line = line.strip('\n').split(' ')
    l.append(line)
print(l)

import cda.ag.linker as lnk
L, supportData = lnk.aprioriRun(l, minSupport= 0.5)
print(L)
print(supportData)

# 计算满足置信度的关联规则
brl = lnk.aprioriRules(L, supportData, minConf= 0.5)
print(brl)