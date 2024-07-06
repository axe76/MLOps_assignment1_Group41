# MLOps Task 3 Docker Containerization

First ```hyperparameter_tuning.py``` is run, which saves the best model.<br>

Run ```docker build . -t sklearn_flask_docker ``` to build docker image.<br>

To test endpoint run
```bash
curl -X PUT http://0.0.0.0:8080/predict/ \
-H 'Content-Type: application/json' \
-d '{"data":[4.98, 2.31, 0.538, 15.3, 6.575, 296, 4.09, 65.2]}'
```
