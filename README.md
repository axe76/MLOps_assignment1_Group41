# MLOps Assignment 1 Group 41

This repo contains all the code related to the Assignment. It is a simple Flask App that triggers inference from a trained model. Model is trained on Boston Housing Dataset.<br>

Curl command for prediction:
```bash
curl -X PUT http://0.0.0.0:8080/predict/ \
-H 'Content-Type: application/json' \
-d '{"data":[4.98, 2.31, 0.538, 15.3, 6.575, 296, 4.09, 65.2]}'
```
