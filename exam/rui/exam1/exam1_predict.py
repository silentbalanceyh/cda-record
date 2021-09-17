import pickle
import warnings

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from exam1 import exam1_func

warnings.filterwarnings('ignore')

base_dir = "data/"
model_dir = "model/"
test_file = 'val_test.csv'

# load data
test_df = pd.read_csv(base_dir + test_file)
print("loaded:", test_df.shape)

# cut_word
# text_cut_list = exam1_func.cut_words(test_df['text'])
# title_cut_list = exam1_func.cut_words(test_df['title'])
with open(model_dir + 'cutwords.file_model', 'rb') as file:
    cutwords_model = pickle.load(file)
    text_cut_list = cutwords_model['text_cut_list_test']
    title_cut_list = cutwords_model['title_cut_list_test']

with open(model_dir + 'cls.file_model', 'rb') as file:
    cls_model = pickle.load(file)
    # logistic_model = cls_model['logistic_model']
    # dtc_model = cls_model['dtc_model']
    svc_model = cls_model['svc_model']
    xgboost_model = cls_model['xgboost_model']
    # rf_model = cls_model['rf_model']
    mlp_model = cls_model['mlp_model']

# assign the best file_model according to file_model scores
best_model = svc_model

# word2vec
word2vec_model = Word2Vec.load(model_dir + 'word_vector.file_model')

# vectorize
textVector_list = exam1_func.vectorize(text_cut_list, word2vec_model)
titleVector_list = exam1_func.vectorize(title_cut_list, word2vec_model)

# predict
X = np.hstack((np.array(textVector_list), np.array(titleVector_list)))
predict_y = best_model.predict(X)
# predict_y_lr = logistic_model.predict(X)
# predict_y_dtc = dtc_model.predict(X)
predict_y_xg = xgboost_model.predict(X)
# predict_y_rf = rf_model.predict(X)
predict_y_mlp = mlp_model.predict(X)

# output
# output = np.vstack((test_df['id'].values, predict_y)).T
# outputDF = pd.DataFrame(output, columns=['id','Predicted_Results'])
output = np.vstack((test_df['id'].values, predict_y, predict_y_xg, predict_y_mlp)).T
outputDF = pd.DataFrame(output, columns=['id','Predicted_Results', 'xg', 'mlp'])
outputDF.to_csv(base_dir + 'results.csv', index=False, sep=',')
