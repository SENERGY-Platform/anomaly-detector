import json 
import tempfile

import unittest
import os
import sys 
print(os.getcwd())
print(sys.path)
from tests._util import setup


class TestOperator(unittest.TestCase):

    def setUp(self):
        with open("algo/test_anomalies/point/opr_config.json") as file:
            self.opr_config = json.load(file)
            self.temp_dir = tempfile.TemporaryDirectory()
            self.opr_config['config']['data_path'] = self.temp_dir.name

        self.mock_kafka_consumer, self.mock_kafka_producer, self.operator = setup(self.opr_config, os.path.join(os.getcwd(), 'algo', 'test_anomalies', 'curve', 'mock_messages.json'),  os.path.join(os.getcwd(), 'mock_re.json'))
    
    def test_1(self):
        while not self.mock_kafka_consumer.empty():
            self.operator._OperatorBase__route()
            print(self.mock_kafka_producer.get_all_outputs())

            
    def tearDown(self):
        self.temp_dir.cleanup()