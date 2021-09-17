import warnings

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import pickle
from sklearn.decomposition import PCA
from collections import Counter


from exam2.fix_outlier import fix_outlier
# custom imports
from exam2.wrong_value_fillna import wrong_value_fillna
from exam2.xgb_fill import xgb_fill

## panda config
pd.options.display.max_columns = None  # 显示所有列
pd.set_option('display.float_format', lambda x: '%.1f' % x)  # 取消科学计数法

warnings.filterwarnings('ignore')

base_dir = "data/"
model_dir = "model/"
train_file = 'val_train.xlsx'

# load data
train_df = pd.read_excel(base_dir + train_file)
print("loaded:", train_df.shape)
print("input y:", Counter(train_df['Target']))
# print(train_df.head(10))
train_df.drop('B0001', inplace=True, axis=1)  # drop id column

# data ut-process step 1, wrong, outlier
print("data ut-process step 1, wrong, outlier")
labelEncoder = LabelEncoder()
y_ = labelEncoder.fit_transform(train_df['Target'])
x_ = train_df.drop('Target', axis=1)  # drop id column
pipe = Pipeline(
        [
         ('wrong_fillna', wrong_value_fillna(wrong_value=['.', '?'])),
         # ('xgb_fillna', xgb_fill()),  # xg is too slow when handling with huge dataset, change to SimpleImputer for fillna
         ('fix', fix_outlier(how='quartile'))
         ])
x_ = pipe.fit_transform(x_, y_)
# data ut-process step 2, fillna, encode for num f_type
print("data ut-process step 2, fillna, encode for num f_type")
features_names = x_.columns.values.tolist()
categorical_features = ['B0002','B0009','B0013','B0019']
numerical_features = [i for i in features_names if i not in categorical_features]
oneHotEncoder = OneHotEncoder(handle_unknown='ignore')
# the transformers should execute exactly follow the order, fillna -> encode
pipe_cat = Pipeline(steps=[('fillna', SimpleImputer(strategy='most_frequent')), # most_frequent, constant fill_value
                         # ('encode', OneHotEncoder()) # move it out to data step 3
                           ])
pipe_num = Pipeline(steps=[('fillna',  SimpleImputer(strategy='median')), # median
                         ('encode', MinMaxScaler()) # StandardScaler, MinMaxScaler
                           ])
preprocessor = ColumnTransformer(transformers=[
    ('cat', pipe_cat, categorical_features),
    ('num', pipe_num, numerical_features)]
)
x_ = preprocessor.fit_transform(x_)

# data ut-process step 3, encode for cat f_type
print("data ut-process step 3, encode for cat f_type")
enc = OneHotEncoder(handle_unknown='ignore')
catColumns = x_[:,0:len(categorical_features)]  # pick up those cat columns
enc.fit(catColumns)
tmpOneHot = enc.transform(catColumns).A # transform to onehot code, and convert to ndarray

# data ut-process step 4, pca on onehot part
pca_model = PCA(n_components=30)
tmpOneHot = pca_model.fit_transform(tmpOneHot)
print("tmpOneHot pca to:", pca_model.n_components_)

# stack pca and rest parts
x_ = x_[:,len(categorical_features):-1] # delete those cat columns with old value
x_ = np.hstack((tmpOneHot, x_)) # add onehot value with rest parts(num f_type columns)

# output
tmp = pd.DataFrame(x_)
tmp['Target'] = y_
tmp.to_csv(base_dir + 'train_processed.csv', encoding='utf-8',index=False, sep=',')

# save onehot file_model
with open(model_dir + 'enc.file_model', 'wb') as file:
    save = {
        'enc' : enc,
    }
    pickle.dump(save, file)


