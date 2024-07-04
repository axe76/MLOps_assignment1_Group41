'''
Flask App
'''

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('trained_model.pkl')
min_max_scaler = joblib.load('min_max_scaler.pkl')

@app.route('/')
def hello():
    '''
    Landing endpoint
    '''
    return '<h1>My Flask App</h1>'

@app.route('/predict/', methods=['PUT'])
def inference():
    '''
    Inference endpoint
    '''
    json_ = request.json
    query = json_['data']
    query = np.array(query).reshape(1,-1)
    input_ = min_max_scaler.transform(query)
    output = model.predict(input_)
    return jsonify({'prediction':list(output)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8080')
