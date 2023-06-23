import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
import pickle

Train = pd.read_csv('./Train.csv')

y_train = Train['Price(euro)']
x_train = Train.drop(columns=['Price(euro)'])

LR = LinearRegression(fit_intercept=True)
LR.fit(x_train, y_train)

scores = cross_validate(LR, x_train, y_train, scoring='r2',
                        cv=ShuffleSplit(n_splits=5, random_state=42))

DF_cv_linreg = pd.DataFrame(scores)
print('Результаты Кросс-валидации', '\n', DF_cv_linreg, '\n')
print('Среднее r2 -', round(DF_cv_linreg.mean()[2], 2))

model = "model.pkl"
with open(model, 'wb') as file:
    pickle.dump(LR, file)
