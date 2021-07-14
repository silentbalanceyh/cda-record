
import pandas as pd
from sklearn.model_selection import train_test_split

base_dir = "data/"
model_dir = "model/"
train_file = 'training_ori.csv'

all_df = pd.read_csv(base_dir + train_file)
print("train loaded:", all_df.shape)

train, test = train_test_split(all_df, test_size=0.2)
result = test[['id','label']]
test = test.drop('label', axis=1)  # drop id column

train.to_csv(base_dir + 'val_train.csv', index=False, sep=',',encoding='utf-8')
test.to_csv(base_dir + 'val_test.csv', index=False, sep=',',encoding='utf-8')
result.to_csv(base_dir + 'val_result.csv', index=False, sep=',',encoding='utf-8')
