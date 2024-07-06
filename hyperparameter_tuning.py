import optuna
from optuna_dashboard import run_server
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

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

def objective(trial):

    params = {
        'alpha': trial.suggest_float('alpha ', 0.0, 2.0, step=0.5),
        'solver': trial.suggest_categorical('solver ', ['svd', 'cholesky']),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'max_iter': trial.suggest_int('max_iter', 500, 2000, step=500),
    }

    model = make_pipeline(PolynomialFeatures(degree=3), linear_model.Ridge(**params))

    model.fit(x_train_scaled, y_train)

    return mean_squared_error(y_test, model.predict(x_test_scaled), squared=False)

storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(storage=storage, direction='minimize')
study.optimize(objective, n_trials=3)

run_server(storage)