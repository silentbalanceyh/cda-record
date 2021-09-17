import pickle
import warnings
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pickle

from exam1 import exam1_func

warnings.filterwarnings('ignore')

base_dir = "data/"
model_dir = "model/"
train_file = 'val_train.csv'

# load data
train_df = pd.read_csv(base_dir + train_file)
print("loaded:", train_df.shape)

# cut_word
# text_cut_list = exam1_func.cut_words(train_df['text'])
# title_cut_list = exam1_func.cut_words(train_df['title'])
with open(model_dir + 'cutwords.file_model', 'rb') as file:
    cutwords_model = pickle.load(file)
    text_cut_list = cutwords_model['text_cut_list']
    title_cut_list = cutwords_model['title_cut_list']

# word2vec
word2vec_model = Word2Vec.load(model_dir + 'word_vector.file_model')

# vectorize
textVector_list = exam1_func.vectorize(text_cut_list, word2vec_model)
titleVector_list = exam1_func.vectorize(title_cut_list, word2vec_model)

# feature process
X = np.hstack((np.array(textVector_list),np.array(titleVector_list)))

labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(train_df['label'])
# split dataset
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

## modeling
# lr
# startTime = time.time()
# logistic_model = LogisticRegression(random_state=0, max_iter=1000, n_jobs=-1)
# # C = [0.01, 0.1, 1.0, 10.0]
# # hyperparameters = dict(C=C)
# # logistic_model = exam1_func.model_func(model_name='lr', file_model=logistic_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
# logistic_model.fit(train_X, train_y)
# print("lr: %s, cost: %.2f" %(logistic_model.score(test_X, test_y), time.time() - startTime))

# dtc
# startTime = time.time()
# dtc_model = DecisionTreeClassifier(random_state=0)
# # max_depth = np.arange(2, 20, 2)
# # hyperparameters = dict(max_depth=max_depth)
# # dtc_model = exam1_func.model_func(model_name='dtc', file_model=dtc_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
# dtc_model.fit(train_X, train_y)
# print("dtc: %s, cost: %.2f" %(dtc_model.score(test_X, test_y), time.time() - startTime))

# xgboost
startTime = time.time()
xgboost_model = xgb.XGBClassifier(n_estimators=1000, eta=0.02, random_state=0, learning_rate=0.01,
                                  n_jobs=-1)  # 'objective': 'multi:softmax' 'num_class': 10,
# max_depth = np.arange(9, 16, 2)
# hyperparameters = dict(max_depth=max_depth)
# xgboost_model = exam1_func.model_func(model_name='xg', file_model=xgboost_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
xgboost_model.fit(train_X, train_y)
print("xg: %s, cost: %.2f" % (xgboost_model.score(test_X, test_y), time.time() - startTime))

# rf
# startTime = time.time()
# rf_model = RandomForestClassifier(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=0)
# # max_depth = np.arange(13, 17, 1)
# # hyperparameters = dict(max_depth=max_depth)
# # rf_model = exam1_func.model_func(model_name='rf', file_model=rf_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
# rf_model.fit(train_X, train_y)
# print("rf: %s, cost: %.2f" %(rf_model.score(test_X, test_y), time.time() - startTime))

# svc
startTime = time.time()
svc_model = SVC(kernel='linear')
# kernel=['rbf','linear','poly','sigmoid']
# hyperparameters = dict(kernel=kernel)
# svc_model = exam1_func.model_func(model_name='svc', file_model=svc_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
svc_model.fit(train_X, train_y)
print("svc: %s, cost: %.2f" % (svc_model.score(test_X, test_y), time.time() - startTime))

# mlp
startTime = time.time()
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), alpha=0.001, max_iter=1000, random_state=0)
# hidden_layer_sizes=[(4,),(8,),(12,)]
# hyperparameters = dict(hidden_layer_sizes=hidden_layer_sizes)
# mlp_model = exam1_func.model_func(model_name='mlp', file_model=mlp_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
mlp_model.fit(train_X, train_y)
print("mlp: %s, cost: %.2f" % (mlp_model.score(test_X, test_y), time.time() - startTime))

# save file_model
with open(model_dir + 'cls.file_model', 'wb') as file:
    save = {
        # 'logistic_model' : logistic_model,
        # 'dtc_model': dtc_model,
        'xgboost_model': xgboost_model,
        # 'rf_model': rf_model,
        'svc_model': svc_model,
        'mlp_model': mlp_model
    }
    pickle.dump(save, file)
