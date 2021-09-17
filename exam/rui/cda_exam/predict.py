import pickle
import warnings

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from cda_exam import func

warnings.filterwarnings('ignore')

base_dir = "data/"
model_dir = "model/"
test_file = 'val_test.xlsx'

# load data
test_df = pd.read_excel(base_dir + test_file)
print("loaded:", test_df.shape)

# cut_word
# text_cut_list = func.cut_words(test_df['message'])
with open(model_dir + 'cutwords.file_model', 'rb') as file:
    cutwords_model = pickle.load(file)
    text_cut_list = cutwords_model['text_cut_list_test']
    # title_cut_list = cutwords_model['title_cut_list_test']

with open(model_dir + 'cls.file_model', 'rb') as file:
    cls_model = pickle.load(file)
    # logistic_model = cls_model['logistic_model']
    # dtc_model = cls_model['dtc_model']
    # svc_model = cls_model['svc_model']
    xgboost_model = cls_model['xgboost_model']
    # rf_model = cls_model['rf_model']
    # mlp_model = cls_model['mlp_model']

# assign the best file_model according to file_model scores
best_model = xgboost_model

# word2vec
word2vec_model = Word2Vec.load(model_dir + 'word_vector.file_model')

# vectorize
textVector_list = func.vectorize(text_cut_list, word2vec_model)
# titleVector_list = func.vectorize(title_cut_list, word2vec_model)

# predict
X = np.array(textVector_list)
predict_y = best_model.predict(X)
# predict_y_lr = logistic_model.predict(X)
# predict_y_dtc = dtc_model.predict(X)
# predict_y_xg = xgboost_model.predict(X)
# predict_y_rf = rf_model.predict(X)
# predict_y_mlp = mlp_model.predict(X)

# output
# output = np.vstack((test_df['id'].values, predict_y)).T
# outputDF = pd.DataFrame(output, columns=['id','Predicted_Results'])
# output = np.vstack((test_df['ID'].values.reshape(-1,1), predict_y)).T
# outputDF = pd.DataFrame(predict_y) #columns=['ID','Predicted_Results']
output = np.hstack((test_df['ID'].values.reshape(-1,1), predict_y))
outputDF = pd.DataFrame(output)
outputDF.to_csv(base_dir + 'results.csv', index=False, sep=',')

