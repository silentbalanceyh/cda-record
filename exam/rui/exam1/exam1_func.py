import time
import warnings
import jieba
import numpy as np
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

# cut words
def cut_words(list):
    # with open(base_dir + 'stopwords.txt', encoding='utf8') as file:
    #     stopWord_list = [k.strip() for k in file.readlines() if k.strip() != '']
    stopWord_list = []
    cutWords_list = []
    i = 0
    startTime = time.time()
    for content in list:
        if isinstance(content, str) and len(content) > 0:
            cutWords = [k for k in jieba.cut(content) if k not in stopWord_list]
            i += 1
            if i % 1000 == 0:
                print('前%d条分词共花费%.2f秒' %(i, time.time()-startTime))
            cutWords_list.append(cutWords)
        else:
            cutWords_list.append([])
    # print(cutWords_list)
    return cutWords_list


# get content vector from word2vec file_model
def get_contentVector(cutWords, word2vec_model):
    dict = word2vec_model.wv.key_to_index
    vector_list = []
    for k in cutWords:
        if k in dict:
            index = dict[k]
            vector = word2vec_model.wv.vectors[index]
            vector_list.append(vector)
    if len(vector_list) > 0:
        contentVector = np.array(vector_list).mean(axis=0)
    else:
        contentVector = np.zeros(100)
    return contentVector


def vectorize(wordList, model):
    startTime = time.time()
    textVector_list = []
    for i in range(len(wordList)):
        cutWords = wordList[i]
        if (i + 1) % 3000 == 0:
            usedTime = time.time() - startTime
            print('前%d篇内容表示成向量共花费%.2f秒' % (i + 1, usedTime))
        textVector_list.append(get_contentVector(cutWords, model))
    return textVector_list


def model_func(model_name, model, param_dict, x_train, y_train):
    hyperparameters = param_dict
    gridsearch = GridSearchCV(model, hyperparameters, cv=5, verbose=1, n_jobs=-1, scoring='f1_micro')
    best_model = gridsearch.fit(x_train, y_train)
    print('best_', model_name, ': ', best_model.best_params_)
    return best_model
