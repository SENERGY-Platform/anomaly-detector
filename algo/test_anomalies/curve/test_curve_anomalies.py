import json 

import algo
import unittest
import os

from tests import MockKafkaProducer, MockKafkaConsumer, init_filter_handler

class TestOperator(unittest.TestCase):

    def test_1(self):
        print(os.getcwd())
        with open("algo/test_anomalies/curve/opr_config.json") as file:
            opr_config = json.load(file)
            mock_kafka_consumer = MockKafkaConsumer(mock_messages)
            mock_kafka_producer = MockKafkaProducer(mock_result)
            operator = algo.Operator()

            operator.init(
                    kafka_consumer=mock_kafka_consumer,
                    kafka_producer=mock_kafka_producer,
                    filter_handler=init_filter_handler(opr_config, None),
                    output_topic=None,
                    pipeline_id=None,
                    operator_id=None
            )
            
            while not mock_kafka_consumer.empty():
                operator._OperatorBase__route()