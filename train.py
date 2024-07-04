from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import pandas as pd
import joblib

data = pd.read_csv('BostonHousing.csv')
print(data.head(5))

column_sels = ['lstat', 'indus', 'nox', 'ptratio', 'rm', 'tax', 'dis', 'age']
x = data.loc[:,column_sels]
y = data['medv']

min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

model = make_pipeline(PolynomialFeatures(degree=3), linear_model.Ridge()).fit(x_scaled, y)
print('Training accuracy: ', model.score(x_scaled,y))
joblib.dump(model, 'trained_model.pkl')
joblib.dump(min_max_scaler, 'min_max_scaler.pkl')