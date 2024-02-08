import threading 
import datetime
import time 
from algo import utils
import json 

import pandas as pd 

# TODO
# nur sinnvoll bei sensoren die regelmaesig senden
# nicht bei erkennungssensoren z.b 

class FrequencyDetector(threading.Thread, utils.StdPointOutlierDetector):
    def __init__(
        self, 
        kafka_producer,
        operator_id,
        pipeline_id,
        output_topic,
        data_path,
        consumer_auto_offset_reset_config
    ):
        threading.Thread.__init__(self)
        utils.StdPointOutlierDetector.__init__(self, data_path)

        self.last_received_ts = None
        self.kafka_producer = kafka_producer
        self.pause_event = threading.Event()
        self.__stop = True
        self.operator_id = operator_id
        self.pipeline_id = pipeline_id
        self.output_topic = output_topic
        self.operator_start_time = datetime.datetime.now()
        self.consumer_auto_offset_reset_config = consumer_auto_offset_reset_config

    def run(self):
        # Frequency Detection shall only run in real time data, not when historic data comes in 
        print(f"Frequency Detector started -> Loop is stopped: {self.__stop}")

        while not self.__stop:
            # TODO remove both checks
            if not self.last_received_ts:
                print("Pause check until first real time input")
                time.sleep(5)
                continue

            if self.last_received_ts < self.operator_start_time:
                print("Last value was historic -> wait until real time data comes in")
                time.sleep(5)
                continue

            now = datetime.datetime.now()
            waiting_time = self.calculate_time_diff(now, self.last_received_ts)
            print(f"Time since last input {waiting_time}")
            anomaly_occured = False
            if self.point_is_anomalous_high(waiting_time):
                sub_type = "high"
                anomaly_occured = True
            elif self.point_is_anomalous_low(waiting_time):
                sub_type = "low"
                anomaly_occured = True

            if anomaly_occured:
                print(f"Anomaly occured: Type=time Sub-Type={sub_type} Value={waiting_time} Mean={self.current_mean} Std={self.current_stddev}")
                self.kafka_producer.produce(
                    self.output_topic,
                        json.dumps(
                            {
                                "pipeline_id": self.pipeline_id,
                                "operator_id": self.operator_id,
                                "analytics": {
                                    "type": "time",
                                    "sub_type": sub_type,
                                    "value": waiting_time,
                                    "unit": "min",
                                },
                                "time": "{}Z".format(datetime.datetime.utcnow().isoformat())
                            }
                        ),
                        self.operator_id
                )
            
            time.sleep(5)

    def stop(self):
        self.__stop = True

    def start_loop(self):
        self.__stop = False 

    def calculate_time_diff(self, ts1, ts2):
        return (ts1 - ts2).total_seconds() / 60

    def register_input(self, input_timestamp): 
        if not self.last_received_ts:
            print('First input arrived')
            self.last_received_ts = input_timestamp
            return 

        waiting_time = self.calculate_time_diff(input_timestamp, self.last_received_ts)
        print(f"Input received at: {input_timestamp} -> Waiting time of current input: {waiting_time} -> Mean={self.current_mean} Std={self.current_stddev}")
        self.update(waiting_time)
        self.last_received_ts = input_timestamp

    
        