import io
import streamlit as st
import gdown
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

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


data_train = train.iloc[:,:3]
scaler  = MinMaxScaler()
scaler.fit_transform(data_train)

data_train = scaler.transform(data_train)
data_test = test.iloc[:,:3]
data_test  = scaler.transform(data_test)

train.iloc[:,:3] = data_train
test.iloc[:,:3] = data_test

train.to_csv('Train.csv', index=False)
test.to_csv('Test.csv', index=False)

y_train = train['Price(euro)']
x_train = train.drop(columns=['Price(euro)'])

LR = LinearRegression(fit_intercept=True)
LR.fit(x_train, y_train)

scores = cross_validate(LR, x_train, y_train, scoring='r2',
                        cv=ShuffleSplit(n_splits=5, random_state=42))

model = "model.pkl"
with open(model, 'wb') as file:
    pickle.dump(LR, file)

y_test = test['Price(euro)']
x_test = test.drop(columns = ['Price(euro)'])

with open('model.pkl', 'rb') as file:
    pickle_model = pickle.load(file)

y_predict = pickle_model.predict(x_test)
# print('r2 для тестовых данных -',  r2_score(y_test,y_predict))
st.title('Результаты работы модели')
result = st.button('Рассчитать Score')
if result:
    DF_cv_linreg = pd.DataFrame(scores)
    st.write('Результаты Кросс-валидации', '\n', DF_cv_linreg, '\n')
    st.write('Среднее r2 -', round(DF_cv_linreg.mean()[2], 2))
