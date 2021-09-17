import matplotlib.pyplot as plt
import pandas as pd
# 读取数据文件
testSet = pd.read_table('data/testSet.txt', header = None)
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 直接切分数据集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    testSet.iloc[:, :-1].values,
    testSet.iloc[:, -1].values,
    random_state=42
)
# 建模
clf = GaussianNB()
clf.fit(Xtrain, Ytrain)
# 直接预测结果
ret = clf.predict(Xtest)
print(ret)
ret1 = clf.predict_proba(Xtest)
print(ret1)
# 计算结果
score = accuracy_score(Ytest, ret)
print(score)

from sklearn import metrics
roc = metrics.roc_curve(Ytest, ret1[:, 1])
print(roc)
fpr, tpr, thresholds = metrics.roc_curve(Ytest, ret1[:,1], drop_intermediate=False)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color = 'darkorange', label = 'ROC curve ( area = %0.2f )' % roc_auc)
plt.plot([0,1],[0,1], color='navy', linestyle = '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc = 'lower right')
plt.show()