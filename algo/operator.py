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
from operator_lib.util.persistence import save, load

from algo.detector import AnomalyDetector

LOG_PREFIX = "MAIN"

def parse_bool(value):
    return (value == "True" or value == "true" or value == "1")

class CustomConfig(Config):
    data_path = "/opt/data"
    check_data_anomalies: bool = False
    check_data_extreme_outlier: bool = True
    check_data_schema: bool = True
    check_receive_time_outlier: bool = True
    check_consumption: bool = False
    init_phase_length: float = 2
    init_phase_level: str = "d"
    train_interval: float = 14
    train_level: str = "d"
    retrain: bool = False
    ml_trainer_url: str = "http://ml-trainer-svc.trainer:5000"
    mlflow_url: str = "http://mlflow-svc.mlflow:5000"
    curve_detector_training_mode: str = "offline"

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        self.check_data_anomalies = parse_bool(self.check_data_anomalies)
        self.check_data_extreme_outlier = parse_bool(self.check_data_extreme_outlier)
        self.check_data_schema = parse_bool(self.check_data_schema)
        self.check_receive_time_outlier = parse_bool(self.check_receive_time_outlier)
        self.retrain = parse_bool(self.retrain)

        if self.init_phase_length != '':
            self.init_phase_length = float(self.init_phase_length)
        else:
            self.init_phase_length = 2
        
        if self.init_phase_level == '':
            self.init_phase_level = 'd'

        if self.train_interval != '':
            self.train_interval = float(self.train_interval)
        else:
            self.train_interval = 14
        
        if self.train_level == '':
            self.train_level = 'd'
        
class Operator(OperatorBase):
    configType = CustomConfig
    
    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)

        if not os.path.exists(self.config.data_path):
            os.mkdir(self.config.data_path)

        #self.produce = lambda x: print(x)  uncomment for local testing to not pollute kafka topics when portforwarding to cluster is used

        self.init_phase_duration = pd.Timedelta(self.config.init_phase_length, self.config.init_phase_level)
        self.operator_start_time = pd.Timestamp(setup_operator_starttime(self.config.data_path)).tz_localize(None)
        self.first_data_time =  load(self.config.data_path, "first_data_time.pickle")
        self.device_type = load(self.config.data_path, "device_type.pickle")
        self.init_median = load(self.config.data_path, "init_median.pickle")
              
        self.init_phase_handler = InitPhase(self.config.data_path, self.init_phase_duration, self.first_data_time, self.produce)
        value = {
            "type": False,
            "sub_type": "",
            "value": "",
            "threshold": 0,
            "mean": 0
        }
        self.init_phase_handler.send_first_init_msg(value)    

        self.device_detectors = {} 
        self.data_list_for_device_type_check = []
        self.device_type = None # Is either "cont_device" or "load_device"
        self.init_median = None # Median value of init phase. Is used as a threshold for checking if device is on or off.

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
            self.produce,
            self.device_type,
            self.init_median,
            self.first_data_time,
            self.config.ml_trainer_url,
            self.config.mlflow_url,
            self.config.curve_detector_training_mode,
            self.get_operator_id(),
            self.get_pipeline_id(),
            self.config.train_level,
            self.config.train_interval,
            self.config.retrain
        )
        self.device_detectors[input_ids] = device_detector
        return device_detector

    def input_is_real_time(self, timestamp):
        return timestamp >= self.operator_start_time
    
    def get_device_type(self):# entries in data_list are of the form (timestamp, data point)
        data_series = pd.Series(data=[data_point for _, data_point in self.data_list_for_device_type_check], index=[timestamp for timestamp, _ in self.data_list_for_device_type_check]).sort_index()
        data_series = data_series[~data_series.index.duplicated(keep='first')]
        device_type = 'cont_device'
        for timestamp_1 in data_series.index:
            constantly_zero = True
            if timestamp_1 + pd.Timedelta(2,'hours') < data_series.index.max():
                for timestamp_2 in data_series.loc[timestamp_1:timestamp_1+pd.Timedelta(2,'hours')].index:
                    if data_series.loc[timestamp_2] > 20: # 20 Watt is a typical bound from above for power demand in standy mode of houshold electricity devices.
                        constantly_zero = False
                        break
                if constantly_zero == True:
                    device_type = 'load_device'
                    break    
        return device_type, data_series.median()

    def run(self, data, selector='energy_func', device_id=''):
        original_input_ids = data.get('original_input_ids')

        input_id = device_id or original_input_ids

        # These operators will also run when historic data is consumed and the init phase is completed based on historic timestamps 
        timestamp = todatetime(data['time']).tz_localize(None)
        value = float(data['value'])

        if not self.first_data_time:
            self.first_data_time = timestamp
            self.init_phase_handler = InitPhase(self.config.data_path, self.init_phase_duration, self.first_data_time, self.produce)

        util.logger.debug(f'{LOG_PREFIX}: Device: {device_id} Input time: {str(data["time"])} Value: {str(data["value"])}')
        
        operator_is_init = self.init_phase_handler.operator_is_in_init_phase(timestamp)
        device_detector = self.get_device_detectors(input_id)
        anomalies_found = None
        timestamp_without_tz = timestamp.tz_localize(None)
        if self.input_is_real_time(timestamp):
            device_detector.start_freq_loop()

        util.logger.debug(f"{LOG_PREFIX}: Check input for anomalies")
        anomalies_found = device_detector.check_input(value, timestamp_without_tz)
        util.logger.debug(f"{LOG_PREFIX}: Found Anomalies: {anomalies_found}")
            
        util.logger.debug(f"{LOG_PREFIX}: Register new input at detectors")
        device_detector.update(value, timestamp_without_tz, self.input_is_real_time(timestamp))
        
        init_value = {
            "type": False,
            "sub_type": "",
            "value": "",
            "threshold": 0,
            "mean": 0
        }
        if operator_is_init:
            self.data_list_for_device_type_check.append((timestamp, value))
            return self.init_phase_handler.generate_init_msg(timestamp, init_value)

        if self.init_phase_handler.init_phase_needs_to_be_reset():
            return self.init_phase_handler.reset_init_phase(init_value)

        if not self.device_type:
            self.device_type, self.init_median = self.get_device_type()
            self.device_detectors[input_id].update_device_type(self.device_type)
            self.device_detectors[input_id].update_init_median(self.init_median)
        
        if anomalies_found and not operator_is_init:
            return anomalies_found
 
    def stop(self):
        super().stop()
        for device, device_detector in self.device_detectors.items():
            util.logger.info(f"Stop Anomaly Detector for device: {device}")
            device_detector.stop()
            util.logger.info("Anomaly Detector stopped")
        # TODO: thread join for frequency detector
        save(self.config.data_path, "first_data_time.pickle", self.first_data_time)
        save(self.config.data_path, "device_type.pickle", self.device_type)
        save(self.config.data_path, "init_median.pickle", self.init_median)
        
