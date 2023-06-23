import pandas as pd
from sklearn.preprocessing import StandardScaler

Train = pd.read_csv('./Train.csv')
Test = pd.read_csv('./Test.csv')

data_train = Train.iloc[:,:3]
scaler  = StandardScaler()
scaler.fit_transform(data_train)
data_train = scaler.transform(data_train)

data_test = Test.iloc[:,:3]
data_test  = scaler.transform(data_test)

Train.iloc[:,:3] = data_train
Train.to_csv('Train.csv', index=False)

Test.iloc[:,:3] = data_test
Test.to_csv('Test.csv', index=False)
