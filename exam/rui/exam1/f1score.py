
import pandas as pd

from sklearn.metrics import f1_score
base_dir = "data/"

val_file = 'val_result.csv'

train_df = pd.read_csv(base_dir + val_file, encoding='utf-8')

print("svm f1 score:", f1_score(train_df['label'],train_df['Predicted_Results'], average='macro'))
print("xg f1 score:", f1_score(train_df['label'],train_df['xg'], average='macro'))
print("mlp f1 score:", f1_score(train_df['label'],train_df['mlp'], average='macro'))

