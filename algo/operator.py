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

__all__ = ("Operator", )

import os
import json 

from algo import utils
import pandas as pd
import datetime
import operator_lib.util as util
from operator_lib.util import Config, OperatorBase, InitPhase, setup_operator_starttime, todatetime, timestamp_to_str

from algo.detector import AnomalyDetector

LOG_PREFIX = "MAIN"

def parse_bool(value):
    return (value == "True" or value == "true" or value == "1" or value)

class CustomConfig(Config):
    data_path = "/opt/data"
    check_data_anomalies: bool = False
    check_data_extreme_outlier: bool = True
    check_data_schema: bool = True
    check_receive_time_outlier: bool = True
    check_consumption: bool = True
    init_phase_length: float = 2
    init_phase_level: str = "d"

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        self.check_data_anomalies = parse_bool(self.check_data_anomalies)
        self.check_data_extreme_outlier = parse_bool(self.check_data_extreme_outlier)
        self.check_data_schema = parse_bool(self.check_data_schema)
        self.check_receive_time_outlier = parse_bool(self.check_receive_time_outlier)
        if self.init_phase_length != '':
            self.init_phase_length = float(self.init_phase_length)
        else:
            self.init_phase_length = 2
        
        if self.init_phase_level == '':
            self.init_phase_level = 'd'
        
class Operator(OperatorBase):
    configType = CustomConfig
    
    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)

        if not os.path.exists(self.config.data_path):
            os.mkdir(self.config.data_path)

        self.init_phase_duration = pd.Timedelta(self.config.init_phase_length, self.config.init_phase_level)
        setup_operator_starttime(self.config.data_path)

        init_phase_duration = pd.Timedelta(self.config.init_phase_length, self.config.init_phase_level)        
        self.init_phase_handler = InitPhase(self.config.data_path, init_phase_duration)
        value = {
            "type": False,
            "sub_type": "",
            "value": "",
            "threshold": 0,
            "mean": 0
        }
        if self.init_phase_handler.first_init_msg_needs_to_send():
            init_msg = self.init_phase_handler.generate_first_init_msg(value)
            self.produce(init_msg)

        self.device_detectors = {} 

    def get_device_detectors(self, input_ids):
        device_detector = self.device_detectors.get(input_ids)
        if device_detector:
            return device_detector
        
        device_detector = AnomalyDetector(
            input_ids,
            self.config.check_receive_time_outlier,
            self.config.check_data_schema,
            self.config.check_data_anomalies,
            self.config.check_data_extreme_outlier,
            self.config.check_consumption,
            self.config.data_path,
            self.produce
        )
        self.device_detectors[input_ids] = device_detector
        return device_detector

    def input_is_real_time(self, timestamp):
        return timestamp >= self.operator_start_time

    def run(self, data, selector='energy_func', device_id=''):
        original_input_ids = data.get('original_input_ids')

        input_id = device_id or original_input_ids

        # These operators will also run when historic data is consumed and the init phase is completed based on historic timestamps 
        timestamp = todatetime(data['time'])
        value = float(data['value'])
        util.logger.debug(f'{LOG_PREFIX}: Device: {device_id} Input time: {str(data["time"])} Value: {str(data["value"])}')

        value = {
            "type": False,
            "sub_type": "",
            "value": "",
            "threshold": 0,
            "mean": 0
        }
        if self.init_phase_handler.operator_is_in_init_phase(timestamp):
            return self.init_phase_handler.generate_init_msg(timestamp, value)
        
        if self.init_phase_handler.init_phase_needs_to_be_reset():
            return self.init_phase_handler.reset_init_phase(value)
 
        device_detector = self.get_device_detectors(input_id)
        anomalies_found = None
        if self.input_is_real_time(timestamp):
            device_detector.start_freq_loop()
            util.logger.debug(f"{LOG_PREFIX}: Check input for anomalies")
            anomalies_found = device_detector.check_input(value, timestamp)
            util.logger.debug(f"{LOG_PREFIX}: Found Anomalies: {anomalies_found}")
            

        util.logger.debug(f"{LOG_PREFIX}: Register new input at detectors")
        device_detector.update(value, timestamp, self.input_is_real_time(timestamp))
        
        if anomalies_found:
            return anomalies_found
 
    def stop(self):
        for device, device_detector in self.device_detectors.items():
            util.logger.info(f"Stop Anomaly Detector for device: {device}")
            device_detector.stop()
            util.logger.info("Anomaly Detector stopped")