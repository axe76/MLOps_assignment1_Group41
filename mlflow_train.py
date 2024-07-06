'''
Sklearn module training
'''

import mlflow
from mlflow.models import infer_signature
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import pandas as pd
import joblib

data = pd.read_csv('BostonHousing.csv')
print(data.head(4))

column_sels = ['lstat', 'indus', 'nox', 'ptratio', 'rm', 'tax', 'dis', 'age']
x = data.loc[:,column_sels]
y = data['medv']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

min_max_scaler = MinMaxScaler()
x_train_scaled = min_max_scaler.fit_transform(X_train)

params = {
    "solver": "auto",
    "max_iter": 1000,
    "positive": True,
    "fit_intercept":False,
    "random_state": 8888,
}

model = make_pipeline(PolynomialFeatures(degree=3), linear_model.Ridge(**params)).fit(x_train_scaled, y_train)
train_preds = model.predict(x_train_scaled)
training_r2_score = model.score(x_train_scaled,y_train)
train_mse = mean_squared_error(y_train, train_preds)
print('Training r2 score: ', training_r2_score)
print('Training mse: ', train_mse)

x_test_scaled = min_max_scaler.transform(X_test)
preds = model.predict(x_test_scaled)
testing_r2_score = model.score(x_test_scaled, y_test)
testing_mse = mean_squared_error(y_test, preds)

print('Test r2 score: ', testing_r2_score)
print('Test mse: ', testing_mse)

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow MLOps Assignment 1")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("train r2 score", training_r2_score)
    mlflow.log_metric("train mse", train_mse)
    mlflow.log_metric("test r2 score", testing_r2_score)
    mlflow.log_metric("test mse", testing_mse)


    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training and Testing Info", "Linear Regression on Boston Housing Dataset")

    # Infer the model signature
    signature = infer_signature(x_train_scaled, model.predict(x_train_scaled))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="boston_housing_lr_model",
        signature=signature,
        input_example=x_train_scaled,
        registered_model_name="lr_ridge_model",
    )
