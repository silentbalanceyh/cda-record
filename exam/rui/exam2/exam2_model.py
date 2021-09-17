import pickle
import time
import warnings

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from exam2 import exam2_func
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import *
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')

base_dir = "data/"
model_dir = "model/"
train_file = 'train_processed.csv'

# load data
train_df = pd.read_csv(base_dir + train_file)

# for OneClassSVM
# X_OneClassSVM = train_df[train_df['Target'] == 0]
# X_OneClassSVM = X_OneClassSVM.drop('Target', axis=1)

labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(train_df['Target'])
X = train_df.drop('Target', axis=1)
X_columns = X.columns

# split dataset
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# SMOTE for over_sampling, only on train dataset
sm = SMOTE(random_state=123)
print("over sampling input:", Counter(train_y))
train_X, train_y = sm.fit_resample(train_X, train_y)
print("over sampling result:", Counter(train_y))

## modeling
# lr
# startTime = time.time()
# logistic_model = LogisticRegression(random_state=0, max_iter=1000, n_jobs=-1,
#                                     # class_weight='balanced' # Imbalance issue
#                                     )
# # C = [0.01, 0.1, 1.0, 10.0]
# # hyperparameters = dict(C=C)
# # logistic_model = exam2_func.model_func(model_name='lr', file_model=logistic_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
# logistic_model.fit(train_X, train_y)
# print("lr: %s, cost: %.2f" %(logistic_model.score(test_X, test_y), time.time() - startTime))

# dtc
# startTime = time.time()
# dtc_model = DecisionTreeClassifier(random_state=0, max_depth=18
#                                    # class_weight='balanced' # Imbalance issue
#                                    )
# # max_depth = np.arange(6, 20, 2)
# # hyperparameters = dict(max_depth=max_depth)
# # dtc_model = exam2_func.model_func(model_name='dtc', file_model=dtc_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
# dtc_model.fit(train_X, train_y)
# print("dtc: %s, cost: %.2f" %(dtc_model.score(test_X, test_y), time.time() - startTime))

# xgboost
startTime = time.time()
xgboost_model = xgb.XGBClassifier(n_estimators=1000, eta=0.02, random_state=0
                                 # scale_pos_weight=100, # Imbalance issue 100:1
                                  )
# max_depth = np.arange(6, 16, 2)
# hyperparameters = dict(max_depth=max_depth)
# xgboost_model = exam2_func.model_func(model_name='xg', file_model=xgboost_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
xgboost_model.fit(train_X, train_y)
print("xg: %s, cost: %.2f" %(xgboost_model.score(test_X, test_y), time.time() - startTime))

# rf
# startTime = time.time()
# rf_model = RandomForestClassifier(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=0, max_depth=16
#                                     # class_weight='balanced' # Imbalance issue
#                                   )
# # max_depth = np.arange(13, 17, 1)
# # hyperparameters = dict(max_depth=max_depth)
# # rf_model = exam2_func.model_func(model_name='rf', file_model=rf_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
# rf_model.fit(train_X, train_y)
# print("rf: %s, cost: %.2f" %(rf_model.score(test_X, test_y), time.time() - startTime))

# xgrf
# startTime = time.time()
# xgrf_model = xgb.XGBRFClassifier(num_parallel_tree=1000,
#                                     # class_weight='balanced' # Imbalance issue
#                                   )
# # max_depth = np.arange(13, 17, 1)
# # hyperparameters = dict(max_depth=max_depth)
# # rf_model = exam2_func.model_func(model_name='rf', file_model=rf_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
# xgrf_model.fit(train_X, train_y)
# print("xgrf: %s, cost: %.2f" %(xgrf_model.score(test_X, test_y), time.time() - startTime))

# irf
# startTime = time.time()
# irf_model = IsolationForest(n_estimators=100, n_jobs=-1, random_state=0,
#                                     # class_weight='balanced' # Imbalance issue
#                                   )
# # max_depth = np.arange(13, 17, 1)
# # hyperparameters = dict(max_depth=max_depth)
# # irf_model = exam2_func.model_func(model_name='irf', file_model=irf_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
# irf_model.fit(X)
# print("irf cost: %.2f" %(time.time() - startTime))

# ocsvm
# startTime = time.time()
# ocsvm_model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1
#                 # class_weight = 'balanced'  # Imbalance issue
#                 )
# # kernel=['rbf','linear','poly','sigmoid']
# # hyperparameters = dict(kernel=kernel)
# # ocsvm_model = exam2_func.model_func(model_name='ocvsm', file_model=ocsvm_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
# ocsvm_model.fit(X_OneClassSVM)
# print("ocsvm cost: %.2f" %(time.time() - startTime))

# svc
startTime = time.time()
svc_model = SVC(kernel='rbf'
                # class_weight = 'balanced'  # Imbalance issue
                )
# kernel=['rbf','linear','poly','sigmoid']
# hyperparameters = dict(kernel=kernel)
# svc_model = exam2_func.model_func(model_name='svc', file_model=svc_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
svc_model.fit(train_X, train_y)
print("svc: %s, cost: %.2f" %(svc_model.score(test_X, test_y), time.time() - startTime))

# mlp
startTime = time.time()
mlp_model = MLPClassifier(hidden_layer_sizes=(100,100), alpha=0.0001, max_iter=1000, random_state=0)
# hidden_layer_sizes=[(4,),(8,),(12,)]
# hyperparameters = dict(hidden_layer_sizes=hidden_layer_sizes)
# mlp_model = exam2_func.model_func(model_name='mlp', file_model=mlp_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
mlp_model.fit(train_X, train_y)
print("mlp: %s, cost: %.2f" %(mlp_model.score(test_X, test_y), time.time() - startTime))

# gaussian NB
# startTime = time.time()
# gnb_model = GaussianNB()
# # hidden_layer_sizes=[(4,),(8,),(12,)]
# # hyperparameters = dict(hidden_layer_sizes=hidden_layer_sizes)
# # mlp_model = exam2_func.model_func(model_name='mlp', file_model=mlp_model, param_dict=hyperparameters, x_train=train_X, y_train=train_y)
# gnb_model.fit(train_X, train_y)
# print("gnb: %s, cost: %.2f" %(gnb_model.score(test_X, test_y), time.time() - startTime))

# save file_model
with open(model_dir + 'cls.file_model', 'wb') as file:
    save = {
        # 'logistic_model' : logistic_model,
        # 'dtc_model': dtc_model,
        'xgboost_model': xgboost_model,
        # 'rf_model': rf_model,
        # 'xgrf_model': xgrf_model,
        # 'irf_model': irf_model,
        # 'ocsvm_model': ocsvm_model,
        'svc_model': svc_model,
        'mlp_model': mlp_model,
        'X_columns': X_columns
    }
    pickle.dump(save, file)
