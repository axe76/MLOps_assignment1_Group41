'''
Dummy deployment script
'''

import ast
import json
import app as deployed_app

def setUp():
    '''
    Initialize flask client
    '''
    deployed_app.app.config['TESTING'] = True
    return deployed_app.app.test_client()

def run_put_endpoint(flask_client):
    '''
    Run inference endpoint
    '''
    headers = {'Content-Type':'application/json'}
    data = {"data":[4.98, 2.31, 0.538, 15.3, 6.575, 296, 4.09, 65.2]}
    r = flask_client.put('/predict/', data=json.dumps(data), headers=headers)
    op = ast.literal_eval(r.data.decode('utf-8'))
    print('Input: ', [4.98, 2.31, 0.538, 15.3, 6.575, 296, 4.09, 65.2])
    print('Inference output', op)

if __name__ == '__main__':
    flask_client = setUp()
    run_put_endpoint(flask_client)
