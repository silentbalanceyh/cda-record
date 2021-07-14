
from exam2 import data_quality_report
import pandas as pd
from sklearn.decomposition import PCA

base_dir = "data/"

check_file = 'train_processed.csv'
#
# train_df = pd.read_excel(base_dir + check_file, encoding='utf-8')
# # print(train_df.describe())
#
# dq = data_quality_report.DQReport(train_df, target='Target', Id='B0001')
# dq.SReport()

# pca test
# train_df = pd.read_csv(base_dir + check_file, encoding='utf-8')
# print(train_df.shape)
# pca = PCA(n_components=0.95)
# train_df = train_df.drop('Target', axis=1)
# pca.fit(train_df)
# print(pca.explained_variance_ratio_)
# # print(pca.explained_variance_)
# print(pca.n_components_)

