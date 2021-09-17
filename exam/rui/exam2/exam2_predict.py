import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from exam2.fix_outlier import fix_outlier
# custom imports
from exam2.wrong_value_fillna import wrong_value_fillna
from exam2.xgb_fill import xgb_fill
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

base_dir = "data/"
model_dir = "model/"
test_file = 'val_test.xlsx'

# load data
test_df = pd.read_excel(base_dir + test_file)
print("loaded:", test_df.shape)
y_ = test_df['B0001']
x_ = test_df.drop('B0001', axis=1)  # drop id column

# load models
with open(model_dir + 'cls.file_model', 'rb') as file:
    cls_model = pickle.load(file)
    # logistic_model = cls_model['logistic_model']
    # dtc_model = cls_model['dtc_model']
    svc_model = cls_model['svc_model']
    xgboost_model = cls_model['xgboost_model']
    # rf_model = cls_model['rf_model']
    # xgrf_model = cls_model['xgrf_model']
    # irf_model = cls_model['irf_model']
    # ocsvm_model = cls_model['ocsvm_model']
    # gnb_model = cls_model['gnb_model']
    mlp_model = cls_model['mlp_model']
    X_columns =  cls_model['X_columns']

with open(model_dir + 'enc.file_model', 'rb') as file:
    oneHot_model = pickle.load(file)
    oneHotEncoder = oneHot_model['enc']


# assign the best file_model according to file_model scores
best_model = xgboost_model

# data ut-process step 1, wrong, outlier
print("data ut-process step 1, wrong, outlier")
pipe = Pipeline(
        [
         ('wrong_fillna', wrong_value_fillna(wrong_value=['.', '?'])),
         # ('xgb_fillna', xgb_fill()),  # xg is too slow when handling with huge dataset, change to SimpleImputer for fillna
         ('fix', fix_outlier(how='quartile'))
         ])
x_ = pipe.fit_transform(x_)

# data ut-process step 2, fillna, encode for num
print("data ut-process step 2, fillna, encode for num f_type")
features_names = x_.columns.values.tolist()
categorical_features = ['B0002','B0009','B0013','B0019']
numerical_features = [i for i in features_names if i not in categorical_features]
# the transformers should execute exactly follow the order, fillna -> encode
pipe_cat = Pipeline(steps=[('fillna', SimpleImputer(strategy='most_frequent')),
                         # ('encode', OneHotEncoder())
                           ])
pipe_num = Pipeline(steps=[('fillna',  SimpleImputer(strategy='median')),
                         ('encode', MinMaxScaler())  # StandardScaler, MinMaxScaler
                           ])
preprocessor = ColumnTransformer(transformers=[
    ('cat', pipe_cat, categorical_features),
    ('num', pipe_num, numerical_features)]
)
X = preprocessor.fit_transform(x_)

# data ut-process step 3, encode for cat f_type
print("data ut-process step 3, encode for cat f_type")
catColumns = X[:,0:len(categorical_features)]
tmpOneHot = oneHotEncoder.transform(catColumns).A

# data ut-process step 4, pca before predicting
pca_model = PCA(n_components=30)
tmpOneHot = pca_model.fit_transform(tmpOneHot)
print("tmpOneHot pca to:", pca_model.n_components_)

# stack pca and rest parts
X = X[:,len(categorical_features):-1]
X = np.hstack((tmpOneHot, X))
X = pd.DataFrame(X, columns=X_columns).astype('float') # set columns order and f_type same as train set


# predict
predict_y = best_model.predict(X)
# predict_y_lr = logistic_model.predict(X)
# predict_y_dtc = dtc_model.predict(X)
# predict_y_xg = xgboost_model.predict(X)
# predict_y_rf = rf_model.predict(X)
# predict_y_irf = irf_model.predict(X)
# predict_y_ocsvm = ocsvm_model.predict(X)
# predict_gnb = gnb_model.predict(X)
predict_y_svc = svc_model.predict(X)
predict_y_mlp = mlp_model.predict(X)


# output
# output = np.vstack((y_.values, predict_y)).T
# outputDF = pd.DataFrame(output, columns=['id','Predicted_Results'])

output = np.vstack((y_.values, predict_y, predict_y_svc, predict_y_mlp)).T
outputDF = pd.DataFrame(output, columns=['id', 'Predicted_Results','svc', 'mlp'])

outputDF.to_csv(base_dir + 'results.csv', index=False, sep=',')
