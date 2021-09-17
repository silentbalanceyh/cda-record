
import pandas as pd

from sklearn.metrics import f1_score
base_dir = "data/"


train_df = pd.read_csv(base_dir + "results.csv", encoding='utf-8')
expect_df = pd.read_csv(base_dir + "val_result.csv", encoding='utf-8')

print("svm f1 score,M:", f1_score(expect_df['label'],train_df['Predicted_Results'], average='macro'))
print("svm f1 score,I:", f1_score(expect_df['label'],train_df['Predicted_Results'], average='micro'))
print("xg f1 score,M:", f1_score(expect_df['label'],train_df['xg'], average='macro'))
print("xg f1 score,I:", f1_score(expect_df['label'],train_df['xg'], average='micro'))
print("mlp f1 score,M:", f1_score(expect_df['label'],train_df['mlp'], average='macro'))
print("mlp f1 score,I:", f1_score(expect_df['label'],train_df['mlp'], average='micro'))

