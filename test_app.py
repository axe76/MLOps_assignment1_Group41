'''
Test Flask App
'''

import unittest
import app as tested_app
import json
import ast


class FlaskAppTests(unittest.TestCase):
    '''
    Class to test endpoints
    '''

    def setUp(self):
        '''
        Initialize flask client
        '''
        tested_app.app.config['TESTING'] = True
        self.app = tested_app.app.test_client()

    def test_inference_endpoint(self):
        '''
        Test inference endpoint
        '''
        headers = {'Content-Type':'application/json'}
        data = {"data":[4.98, 2.31, 0.538, 15.3, 6.575, 296, 4.09, 65.2]}
        r = self.app.put('/predict/', data=json.dumps(data), headers=headers)
        op = ast.literal_eval(r.data.decode('utf-8'))
        print(op)
        self.assertTrue(len(op['prediction'])>0)

if __name__ == '__main__':
    unittest.main()
