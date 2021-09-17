import time
import warnings
import jieba
import numpy as np
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')


def model_func(model_name, model, param_dict, x_train, y_train):
    hyperparameters = param_dict
    gridsearch = GridSearchCV(model, hyperparameters, verbose=1, n_jobs=-1, scoring='f1_macro')
    best_model = gridsearch.fit(x_train, y_train)
    print('best_', model_name, ': ', best_model.best_params_)
    return best_model