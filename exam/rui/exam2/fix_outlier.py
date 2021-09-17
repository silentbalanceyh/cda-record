from sklearn.base import BaseEstimator, TransformerMixin
from exam2.common_functions import get_kind

class fix_outlier(BaseEstimator, TransformerMixin):
    def __init__(self,
                 num_list=None,
                 diff_num: int = 8,
                 pmin: float = None,
                 pmax: float = None,
                 how: str = 'quartile'):
        self.num_list = num_list
        self.diff_num = diff_num
        self.pmin = pmin
        self.pmax = pmax
        self.how = how

    def fit(self, X, y=None):
        # print("fit fix_outlier")
        X = X.copy()
        if self.num_list is None:
            self.num_list = []
            for col in X.columns:
                kind = get_kind(x=X[col], diff_limit=self.diff_num)
                if kind == 'numeric':
                    self.num_list.append(col)
        return self

    def transform(self, X, y=None):
        # print("transform fix_outlier")
        X = X.copy()
        for col in self.num_list:
            describe_ = X[col].describe()
            fmin_, fmax_ = 0.0, 0.0
            if self.how == 'quartile':
                IQR = round(describe_['75%'] - describe_['25%'], 2)
                fmin_ = round(describe_['25%'] - 1.5 * IQR, 2)
                fmax_ = round(describe_['75%'] + 1.5 * IQR, 2)
            elif self.how == 'self_percent':
                fmin_ = round(X[col].quantile(self.pmin), 2)
                fmax_ = round(X[col].quantile(self.pmax), 2)
            elif self.how == 'cap':
                fmin_ = round(describe_['mean'] - 3 * describe_['std'], 2)
                fmax_ = round(describe_['mean'] + 3 * describe_['std'], 2)
            else:
                print("don't have that method!")

            # print('deal with "' + col + '" lower fmin size: ' + str(X.loc[X[col] < fmin_, col].shape[0]))
            # print('deal with "' + col + '" higher fmax size: ' + str(X.loc[X[col] > fmax_, col].shape[0]))
            X.loc[X[col] < fmin_, col] = fmin_
            X.loc[X[col] > fmax_, col] = fmax_
        print('fix outlier success!')
        return X