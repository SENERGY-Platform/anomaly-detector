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
from operator_lib.util import Config
from operator_lib.util import OperatorBase

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
        self.setup_operator_start(self.config.data_path)
        self.init_phase_resetted = utils.load_init_phase_was_resetted(self.config.data_path)
        self.send_init_message()

        self.setup_device_detectors()

    def setup_device_detectors(self):
        self.device_detectors = {} 
        dep_config = util.DeploymentConfig()
        config_json = json.loads(dep_config.config)
        opr_config = util.OperatorConfig(config_json)
        # Input can come from one/multiple service and one/multiple devices 
        for input_topic in opr_config.inputTopics:
            device_ids = input_topic.filterValue
            for device_id in device_ids.split(','):
                util.logger.info(f"Initialize Detector for Device: {device_id}")
                self.device_detectors[device_id] = AnomalyDetector(
                    device_id,
                    self.config.check_receive_time_outlier,
                    self.config.check_data_schema,
                    self.config.check_data_anomalies,
                    self.config.check_data_extreme_outlier,
                    self.config.check_consumption,
                    self.config.data_path,
                    self.produce
                )

    def setup_operator_start(self, data_path):
        self.operator_start_time = utils.load_operator_start_time(data_path)
        if not self.operator_start_time:
            self.operator_start_time = datetime.datetime.now()
            util.logger.info(f"Store operator start time not found -> create and save")
            utils.save_operator_start_time(data_path, self.operator_start_time)
        util.logger.info(f"Operator start time: {self.operator_start_time}")

    def input_is_real_time(self, timestamp):
        return timestamp >= self.operator_start_time

    def operator_is_in_init_phase(self, timestamp):
        return timestamp-self.operator_start_time < self.init_phase_duration

    def generate_init_message(self, minutes_until_start=None):
        if not minutes_until_start:
            minutes_until_start = int(self.init_phase_duration.total_seconds()/60)

        return {
                "type": "",
                "sub_type": "",
                "value": "",
                "threshold": 0,
                "mean": 0,
                "initial_phase": f"Die Anwendung befindet sich noch für ca. {minutes_until_start} Minuten in der Initialisierungsphase"
            }

    def send_init_message(self):
        self.produce(self.generate_init_message())

    def reset_init_phase(self):
        self.produce({
                "type": "",
                "sub_type": "",
                "value": "",
                "threshold": 0,
                "mean": 0,
                "initial_phase": ""
        })
        self.init_phase_resetted = True
        utils.save_init_phase_was_resetted(self.config.data_path, True)

    def handle_init_phase(self, timestamp):
        # Use input timestamp and first input for historic and real time data support 
        if self.operator_is_in_init_phase(timestamp):
            util.logger.debug(f"{LOG_PREFIX}: Still in initialisation phase! {timestamp} - {self.operator_start_time} < {self.init_phase_duration}")
            td_until_start = self.init_phase_duration - (timestamp - self.operator_start_time)
            minutes_until_start = int(td_until_start.total_seconds()/60)
            return self.generate_init_message(minutes_until_start)

        if not self.input_is_real_time(timestamp) or self.init_phase_resetted:
            return None 

        util.logger.debug(f"{LOG_PREFIX}: Reset init phase message")
        self.reset_init_phase()

    def run(self, data, selector='energy_func', device_id=''):
        # These operators will also run when historic data is consumed and the init phase is completed based on historic timestamps 
        timestamp = utils.todatetime(data['time']).tz_localize(None)
        value = float(data['value'])
        util.logger.debug(f'{LOG_PREFIX}: Device: {device_id} Input time: {str(timestamp)} Value: {str(data["value"])}')

        device_detector = self.device_detectors[device_id]
        if not self.operator_is_in_init_phase(timestamp) and self.input_is_real_time(timestamp):
            util.logger.debug(f"{LOG_PREFIX}: Check input for anomalies")
            anomalies_found = device_detector.check_input(value, timestamp)
            util.logger.debug(f"{LOG_PREFIX}: Found Anomalies: {anomalies_found}")
            if anomalies_found:
                return anomalies_found

        util.logger.debug(f"{LOG_PREFIX}: Register new input at detectors")
        device_detector.update(value, timestamp, self.input_is_real_time(timestamp))

        if not self.operator_is_in_init_phase(timestamp):
            util.logger.debug(f"{LOG_PREFIX}: Start Frequency Detector Loop")
            device_detector.start_freq_loop()

        init_msg = self.handle_init_phase(timestamp)
        if init_msg:
            return init_msg

    def stop(self):
        for device, device_detector in self.device_detectors.items():
            util.logger.info(f"Stop Anomaly Detector for device: {device}")
            device_detector.stop()
            util.logger.info("Anomaly Detector stopped")