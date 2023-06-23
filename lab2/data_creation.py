import gdown
import pandas as pd
from sklearn.model_selection import train_test_split


id = "1xOEOvSdBXPXXEcWV7VtiS7RoyEuDYzLC"
output = "cars_moldova.csv"
gdown.download(quiet=False, id=id, output=output)

df = pd.read_csv('./cars_moldova.csv')

num_columns = []
for column_name in df.columns:
    if (df[column_name].dtypes != object):
        num_columns +=[column_name]

df_num = df[num_columns].copy()
train, test = train_test_split(df_num, test_size=0.3, random_state=42)

train.to_csv('Train.csv', index=False)
test.to_csv('Test.csv', index=False)
