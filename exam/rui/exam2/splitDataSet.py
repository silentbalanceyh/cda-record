
import pandas as pd
from sklearn.model_selection import train_test_split

base_dir = "data/"
model_dir = "model/"
train_file = 'Train_all.xlsx'

all_df = pd.read_excel(base_dir + train_file)
print("train loaded:", all_df.shape)

train, test = train_test_split(all_df, test_size=0.2)
result = test[['B0001','Target']]
test = test.drop('Target', axis=1)  # drop id column

train.to_excel(base_dir + 'val_train.xlsx',encoding='utf-8', index=False)
test.to_excel(base_dir + 'val_test.xlsx',encoding='utf-8', index=False)
result.to_excel(base_dir + 'val_result.xlsx',encoding='utf-8', index=False)



# train = train_df[0:13000]
# train.to_csv(base_dir + 'val_train.csv', index=False, sep=',')
#
# test = train_df[13000:-1]
# result = test[['id','label']]
# result.to_csv(base_dir + 'val_result.csv', index=False, sep=',')
# test = test.drop('label', axis=1)  # drop id column
# test.to_csv(base_dir + 'val_test.csv', index=False, sep=',')
