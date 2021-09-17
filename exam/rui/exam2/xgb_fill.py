import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from exam2.common_functions import get_kind


class xgb_fill(BaseEstimator, TransformerMixin):
    def __init__(self,
                 num_list: list = None,
                 cate_list: list = None,
                 diff_num: int = 10,
                 include_y: bool = False,
                 random_state: int = 0):
        self.num_list = num_list
        self.cate_list = cate_list
        self.diff_num = diff_num
        self.include_y = include_y
        self.random_state = random_state

        self.xgb_cla_dict = {}
        self.xgb_reg_dict = {}
        self.y = None

    def fit(self, X, y=None):
        from tqdm import tqdm
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

        if self.include_y:
            self.y = y
            X['y'] = self.y

        for col in tqdm(self.cate_list):
            file = X.copy()
            if file[col].isnull().any():
                print('xg fitting:', col)
                df = pd.get_dummies(file, columns=[i for i in self.cate_list if i != col],
                                    prefix=[i for i in self.cate_list if i != col],
                                    dummy_na=True)
                not_null = df.dropna(subset=[col])
                # null = df.drop(not_null.index)
                x_ = not_null.drop([col], axis=1)
                y_ = not_null[col]
                xgb_cla = xgb.XGBClassifier(random_state=self.random_state)
                xgb_cla.fit(x_, y_)
                # null[col] = xgb_cla.predict(null.drop([col], axis=1))
                self.xgb_cla_dict[col] = xgb_cla

        for col in tqdm(self.num_list):
            file = X.copy()
            if file[col].isnull().any():
                print('xg fitting:', col)
                df = pd.get_dummies(file, columns=self.cate_list, dummy_na=True, prefix=self.cate_list)
                not_null = df.dropna(subset=[col])
                # null = df.drop(not_null.index)
                x_ = not_null.drop([col], axis=1)
                y_ = not_null[col]
                xgb_reg = xgb.XGBRegressor(random_state=self.random_state, objective='reg:squarederror')
                xgb_reg.fit(x_, y_)
                # null[col] = xgb_reg.predict(null.drop([col], axis=1))
                self.xgb_reg_dict[col] = xgb_reg
        return self

    def transform(self, X):
        X = X.copy()
        # print("transform xgb")
        if self.include_y:
            X['y'] = self.y
        from tqdm import tqdm
        for col in tqdm(self.cate_list):
            file = X.copy()
            if file[col].isnull().any():
                print('xg filling:', col)
                df = pd.get_dummies(file, columns=[i for i in self.cate_list if i != col],
                                    prefix=[i for i in self.cate_list if i != col],
                                    dummy_na=True)
                not_null = df.dropna(subset=[col])
                null = df.drop(not_null.index)
                null[col] = self.xgb_cla_dict[col].predict(null.drop([col], axis=1))
                X[col] = pd.concat([null, not_null], axis=0)[col]
            else:
                X[col] = file[col]

        for col in tqdm(self.num_list):
            file = X.copy()
            if file[col].isnull().any():
                print('xg filling:', col)
                df = pd.get_dummies(file, columns=self.cate_list, dummy_na=True, prefix=self.cate_list)
                not_null = df.dropna(subset=[col])
                null = df.drop(not_null.index)
                null[col] = self.xgb_reg_dict[col].predict(null.drop([col], axis=1))
                null[col] = null[col].round()
                X[col] = pd.concat([null, not_null], axis=0)[col]
            else:
                X[col] = file[col]
        if self.include_y:
            X.drop('y', axis=1, inplace=True)
        return X