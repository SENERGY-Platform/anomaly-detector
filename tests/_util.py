"""
   Copyright 2022 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import mf_lib
import logging
import json
import queue
import operator_lib.util as util

import algo 
logger = logging.getLogger("operator")
logger.disabled = True


def init_filter_handler(opr_config, pipeline_id: str):
    if not isinstance(opr_config, util.OperatorConfig):
        opr_config = util.OperatorConfig(opr_config)
    filter_handler = mf_lib.FilterHandler()
    for it in opr_config.inputTopics:
        filter_handler.add_filter(util.gen_filter(input_topic=it, pipeline_id=pipeline_id))
    return filter_handler


class MockOperator(util.OperatorBase):
    def func_1(self, a, timestamp):
        assert a == mock_messages[0]["data"]["val_a"]
        assert timestamp == mock_messages[0]["data"]["time"]
        return {"result": 1}

    def func_2(self, a, b, timestamp):
        assert a == mock_messages[1]["data"]["val_a"]
        assert b == mock_messages[1]["data"]["val_b"]
        assert timestamp == mock_messages[1]["data"]["time"]

    def run(self, data, selector):
        return getattr(self, selector)(**data)


class MockKafkaMessage:
    def __init__(self, value=None, err_obj=None):
        self.__value = value
        self.__err_obj = err_obj

    def error(self):
        return self.__err_obj

    def value(self):
        return self.__value


class MockKafkaConsumer:
    def __init__(self, messages):
        self.__queue = queue.Queue()
        for m in messages:
            self.__queue.put(MockKafkaMessage(json.dumps(m)))

    def poll(self, timeout=1.0):
        try:
            return self.__queue.get(timeout=timeout)
        except queue.Empty:
            pass

    def empty(self):
        return self.__queue.empty()


class MockKafkaProducer:
    def __init__(self, result):
        self.__result = result
        self.__count = 0

    def produce(self, topic, value, key):
        assert self.__count < 1
        assert topic == self.__result["topic"]
        assert key == self.__result["key"]
        assert isinstance(value, str)
        value = json.loads(value)
        assert set(value) == set(self.__result["value"])
        assert value["pipeline_id"] == self.__result["value"]["pipeline_id"]
        assert value["operator_id"] == self.__result["value"]["operator_id"]
        assert isinstance(value["analytics"], dict)
        assert isinstance(value["time"], str)
        self.__count += 1

class MockKafkaCollectProducer:
    def __init__(self):
        self.outputs = []

    def produce(self, topic, value, key):
        self.outputs.append({
            'topic': topic,
            'value': value,
            'key': key
        })
    
    def get_all_outputs(self):
        return self.outputs

    def get_last_output(self):
        return self.outputs[-1]

def setup(opr_config, path_to_expected_input_messages, path_to_expected_output):
    with open(path_to_expected_input_messages, "r") as msg_file:
        mock_messages = json.loads(msg_file.read())
        mock_kafka_consumer = MockKafkaConsumer(mock_messages)
    
    mock_kafka_producer = MockKafkaCollectProducer()
    
    operator = algo.Operator()

    operator.init(
            kafka_consumer=mock_kafka_consumer,
            kafka_producer=mock_kafka_producer,
            filter_handler=init_filter_handler(opr_config, None),
            output_topic=None,
            pipeline_id=None,
            operator_id=None,
            config=operator.configType(opr_config['config'])
    )
    return mock_kafka_consumer, mock_kafka_producer, operator