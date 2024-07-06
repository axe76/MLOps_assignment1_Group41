import optuna
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib

data = pd.read_csv('BostonHousing.csv')
# print(data.head(4))

column_sels = ['lstat', 'indus', 'nox', 'ptratio', 'rm', 'tax', 'dis', 'age']
x = data.loc[:,column_sels]
y = data['medv']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

min_max_scaler = MinMaxScaler()
x_train_scaled = min_max_scaler.fit_transform(X_train)
x_test_scaled = min_max_scaler.transform(X_test)

joblib.dump(min_max_scaler, 'min_max_scaler.pkl')

best_model = None
global_model = None

def objective(trial):
    global global_model
    params = {
        'alpha': trial.suggest_float('alpha ', 0.0, 2.0, step=0.5),
        'solver': trial.suggest_categorical('solver ', ['svd', 'cholesky']),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'max_iter': trial.suggest_int('max_iter', 500, 2000, step=500),
    }

    model = make_pipeline(PolynomialFeatures(degree=3), linear_model.Ridge(**params))

    model.fit(x_train_scaled, y_train)

    global_model = model

    return mean_squared_error(y_test, model.predict(x_test_scaled), squared=False)

def callback(study, trial):
    global best_model
    if study.best_trial == trial:
        best_model = global_model

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=3, callbacks=[callback])

joblib.dump(best_model, 'best_model.pkl')