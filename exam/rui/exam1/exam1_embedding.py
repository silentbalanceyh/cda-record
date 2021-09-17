import time
import warnings
import pandas as pd
from gensim.models import Word2Vec
from exam1 import exam1_func
from collections import Counter
import pickle

warnings.filterwarnings('ignore')

base_dir = "data/"
model_dir = "model/"
train_file = 'val_train.csv'
test_file = 'val_test.csv'

# load data
train_df = pd.read_csv(base_dir + train_file)
print("train loaded:", train_df.shape)
print("input y:", Counter(train_df['label']))
test_df = pd.read_csv(base_dir + test_file)
print("test loaded:", test_df.shape)

# cut_word
text_cut_list = exam1_func.cut_words(train_df['text'])
title_cut_list = exam1_func.cut_words(train_df['title'])
text_cut_list_test = exam1_func.cut_words(test_df['text'])
title_cut_list_test = exam1_func.cut_words(test_df['title'])

# word2vec
startTime = time.time()
word2vec_model = Word2Vec(text_cut_list + title_cut_list, size=100, iter=5, min_count=1) # text_cut_list + title_cut_list+text_cut_list_test+title_cut_list_test
usedTime = time.time() - startTime
print('形成word2vec模型共花费%.2f秒' %usedTime)
word2vec_model.save(model_dir + 'word_vector.file_model')

# save cutwords file_model
with open(model_dir + 'cutwords.file_model', 'wb') as file:
    save = {
        'text_cut_list' : text_cut_list,
        'title_cut_list': title_cut_list,
        'text_cut_list_test': text_cut_list_test,
        'title_cut_list_test': title_cut_list_test,
    }
    pickle.dump(save, file)