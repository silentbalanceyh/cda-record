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
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from cda_exam import func

from cda_exam import func

warnings.filterwarnings('ignore')

base_dir = "data/"
model_dir = "model/"
train_file = 'training.xlsx'

# load data
train_df = pd.read_excel(base_dir + train_file,  encoding='utf-8')
print("loaded:", train_df.shape)

# cut_word
text_cut_list = func.cut_words(train_df['message'])
# with open(model_dir + 'cutwords.model', 'rb') as file:
#     cutwords_model = pickle.load(file)
#     text_cut_list = cutwords_model['text_cut_list']

# word2vec
word2vec_model = Word2Vec.load(model_dir + 'word_vector.model')

# vectorize
textVector_list = func.vectorize(text_cut_list, word2vec_model)

# feature process
X = np.array((textVector_list))

# multi class y
# labelEncoder = LabelEncoder()
# y = labelEncoder.fit_transform(train_df['target'])
y = np.vstack((train_df['none'], train_df['disgust'],  train_df['like'], train_df['happiness'], train_df['sadness'], train_df['surprise'], train_df['anger'], train_df['fear'])).T

# split dataset
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

## modeling
# lr
# startTime = time.time()
# logistic_model = LogisticRegression(random_state=0, max_iter=1000, n_jobs=-1)
# # C = [0.01, 0.1, 1.0, 10.0]
# # hyperparameters = dict(C=C)
# # logistic_model = func.model_func(model_name='lr', model=logistic_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
# logistic_model.fit(train_X, train_y)
# print("lr: %s, cost: %.2f" %(logistic_model.score(test_X, test_y), time.time() - startTime))

# dtc
# startTime = time.time()
# dtc_model = DecisionTreeClassifier(random_state=0)
# # max_depth = np.arange(2, 20, 2)
# # hyperparameters = dict(max_depth=max_depth)
# # dtc_model = func.model_func(model_name='dtc', model=dtc_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
# dtc_model.fit(train_X, train_y)
# print("dtc: %s, cost: %.2f" %(dtc_model.score(test_X, test_y), time.time() - startTime))

# xgboost
startTime = time.time()
xgboost_model = xgb.XGBClassifier(n_estimators=1000, eta=0.02, random_state=0, learning_rate=0.01 ,
                                  n_jobs=-1) # 'objective': 'multi:softmax' 'num_class': 10,
or_xg_model = xgboost_model
# # max_depth = np.arange(9, 16, 2)
# # hyperparameters = dict(max_depth=max_depth)
# # xgboost_model = func.model_func(model_name='xg', model=xgboost_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
# xgboost_model.fit(train_X, train_y)
# print("xg: %s, cost: %.2f" %(xgboost_model.score(test_X, test_y), time.time() - startTime))
# # multi categories, one vs rest
or_xg_model = OneVsRestClassifier(or_xg_model, n_jobs=-1)
or_xg_model.fit(train_X, train_y)
print("or_xg: %s, cost: %.2f" %(or_xg_model.score(test_X, test_y), time.time() - startTime))

# rf
# startTime = time.time()
# rf_model = RandomForestClassifier(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=0)
# # max_depth = np.arange(13, 17, 1)
# # hyperparameters = dict(max_depth=max_depth)
# # rf_model = func.model_func(model_name='rf', model=rf_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
# rf_model.fit(train_X, train_y)
# print("rf: %s, cost: %.2f" %(rf_model.score(test_X, test_y), time.time() - startTime))

# svc
# startTime = time.time()
# svc_model = SVC(kernel='linear')
# # kernel=['rbf','linear','poly','sigmoid']
# # hyperparameters = dict(kernel=kernel)
# # svc_model = func.model_func(model_name='svc', model=svc_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
# # svc_model = OneVsRestClassifier(svc_model, n_jobs=-1)
# svc_model.fit(train_X, train_y)
# print("svc: %s, cost: %.2f" %(svc_model.score(test_X, test_y), time.time() - startTime))

# mlp
# startTime = time.time()
# mlp_model = MLPClassifier(hidden_layer_sizes=(100,), alpha=0.001, max_iter=1000, random_state=0)
# or_mlp_model = mlp_model
# # hidden_layer_sizes=[(4,),(8,),(12,)]
# # hyperparameters = dict(hidden_layer_sizes=hidden_layer_sizes)
# # mlp_model = func.model_func(model_name='mlp', model=mlp_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
# mlp_model.fit(train_X, train_y)
# print("mlp: %s, cost: %.2f" %(mlp_model.score(test_X, test_y), time.time() - startTime))
# # multi categories, one vs rest
# or_mlp_model = OneVsRestClassifier(or_mlp_model, n_jobs=-1)
# or_mlp_model.fit(train_X, train_y)
# print("or_mlp: %s, cost: %.2f" %(or_mlp_model.score(test_X, test_y), time.time() - startTime))

# save model
with open(model_dir + 'cls.model', 'wb') as file:
    save = {
        # 'logistic_model' : logistic_model,
        # 'dtc_model': dtc_model,
        'xgboost_model': or_xg_model,
        # 'rf_model': rf_model,
        # 'svc_model': svc_model,
        # 'mlp_model': mlp_model
    }
    pickle.dump(save, file)
