import pickle
import time
import warnings

import pandas as pd
from cda_exam import func
from gensim.models import Word2Vec

warnings.filterwarnings('ignore')

base_dir = "data/"
model_dir = "model/"
train_file = 'training.xlsx'
test_file = 'test.xlsx'

# load data
train_df = pd.read_excel(base_dir + train_file, encoding='utf-8')
print("train loaded:", train_df.shape)
test_df = pd.read_excel(base_dir + test_file, encoding='utf-8')
print("test loaded:", test_df.shape)
# print("input y:", Counter(train_df['label']))
# test_df = pd.read_csv(base_dir + test_file)
# print("test loaded:", test_df.shape)

# cut_word
text_cut_list = func.cut_words(train_df['message'])
# text_cut_list_test = exam1_func.cut_words(test_df['text'])
text_cut_list_test = func.cut_words(test_df['message'])

# word2vec
startTime = time.time()
word2vec_model = Word2Vec(text_cut_list + text_cut_list_test, size=200, iter=5,
                          min_count=1)  # text_cut_list + title_cut_list+text_cut_list_test+title_cut_list_test
usedTime = time.time() - startTime
print('形成word2vec模型共花费%.2f秒' % usedTime)
word2vec_model.save(model_dir + 'word_vector.model')

# save cutwords model
with open(model_dir + 'cutwords.model', 'wb') as file:
    save = {
        'text_cut_list': text_cut_list,
        'text_cut_list_test': text_cut_list_test,
    }
    pickle.dump(save, file)
