from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Generate a random multilabel classification problem.
# For each sample, the generative process is:
#     pick the number of labels: n ~ Poisson(n_labels)
#     n times, choose a class c: c ~ Multinomial(theta)
#     pick the document length: k ~ Poisson(length)
#k times, choose a word: w ~ Multinomial(theta_c)

X, Y = make_multilabel_classification(n_samples=10, n_features=5, n_classes=3, n_labels=2)

# Split dataset to 8:2
X_train, X_test, Y_train ,Y_test = train_test_split(X, Y, test_size=0.2)

cls = DecisionTreeClassifier()
cls.fit(X_train, Y_train)
print("rf: %s" %(cls.score(X_test, Y_test)))

