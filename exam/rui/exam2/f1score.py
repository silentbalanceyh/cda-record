
import pandas as pd

from sklearn.metrics import f1_score
base_dir = "data/"

val_file = 'val_result.xlsx'

train_df = pd.read_excel(base_dir + val_file)
train_pr = pd.read_csv(base_dir + "results.csv")

# print("svm f1 score:", f1_score(train_df['Target'],train_df['Predicted_Results'], average='weighted'))
# print("lr f1 score:", f1_score(train_df['Target'],train_df['lr'], average='weighted'))
# print("dtc f1 score:", f1_score(train_df['Target'],train_df['dtc'], average='weighted'))
print("xg f1 score macro:", f1_score(train_df['Target'],train_pr['Predicted_Results'], average='macro'))
print("xg f1 score micro:", f1_score(train_df['Target'],train_pr['Predicted_Results'], average='micro'))
# print("rf f1 score:", f1_score(train_df['Target'],train_df['rf'], average='weighted'))
# print("mlp f1 score:", f1_score(train_df['Target'],train_df['mlp'], average='weighted'))
# print("irf f1 score macro:", f1_score(train_df['Target'],train_df['irf'], average='macro'))
# print("ocsvm f1 score macro:", f1_score(train_df['Target'],train_df['ocsvm'], average='macro'))

print("svc f1 score macro:", f1_score(train_df['Target'],train_pr['svc'], average='macro'))
print("svc f1 score micro:", f1_score(train_df['Target'],train_pr['svc'], average='micro'))
print("mlp f1 score macro:", f1_score(train_df['Target'],train_pr['mlp'], average='macro'))
print("svc f1 score micro:", f1_score(train_df['Target'],train_pr['svc'], average='micro'))



