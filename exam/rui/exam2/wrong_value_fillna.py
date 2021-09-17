import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from exam2.common_functions import get_kind


class wrong_value_fillna(BaseEstimator, TransformerMixin):
    def __init__(self,
                 num_list: list = None,
                 cate_list: list = None,
                 wrong_value: list = None,
                 diff_num: int = 8):
        self.num_list = num_list
        self.cate_list = cate_list
        self.diff_num = diff_num
        self.wrong_value = wrong_value

    def fit(self, X, y=None):
        X = X.copy()
        if self.num_list is None:
            self.num_list = []
            for col in X.columns:
                kind = get_kind(x=X[col], diff_limit=self.diff_num)
                if kind == 'numeric':
                    self.num_list.append(col)

        if self.cate_list is None:
            self.cate_list = []
            for col in X.columns:
                kind = get_kind(x=X[col], diff_limit=self.diff_num)
                if kind == 'categorical':
                    self.cate_list.append(col)
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.replace(self.wrong_value, np.nan, inplace=True)
        for col in X.columns:
            if get_kind(X[col]) == 'numeric':
                X[col] = X[col].astype('float')
            else:
                X[col] = X[col].astype('object')
        print('wrong_value_fillna success!')
        return X