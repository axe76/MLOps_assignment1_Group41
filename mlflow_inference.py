import mlflow
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import pandas as pd
import joblib

data = pd.read_csv('BostonHousing.csv')

column_sels = ['lstat', 'indus', 'nox', 'ptratio', 'rm', 'tax', 'dis', 'age']
x = data.loc[:,column_sels]
y = data['medv']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

min_max_scaler = MinMaxScaler()
min_max_scaler = min_max_scaler.fit(X_train)

x_test_scaled = min_max_scaler.transform(X_test)

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

logged_model = 'runs:/9895db617a1d4c04ba6353572a6216aa/boston_housing_lr_model'

loaded_model = mlflow.pyfunc.load_model(logged_model)

predictions = loaded_model.predict(x_test_scaled)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print('Test r2 score: ', r2)
print('Test MSE: ', mse)