import pandas as pd
import pickle
from sklearn.metrics import r2_score

Test = pd.read_csv('./Test.csv')

y_test = Test['Price(euro)']
x_test = Test.drop(columns = ['Price(euro)'])

with open('model.pkl', 'rb') as file:
    pickle_model = pickle.load(file)

y_predict = pickle_model.predict(x_test)
print('r2 для тестовых данных -',  r2_score(y_test,y_predict))
