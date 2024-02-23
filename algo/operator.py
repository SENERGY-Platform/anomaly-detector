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

from algo import curve_anomaly
from algo import point_outlier
from algo import consumption_anomaly
from algo import utils
from algo.frequency_point_outlier import FrequencyDetector
import pandas as pd
import datetime

from operator_lib.util import Config
from operator_lib.util import OperatorBase

def parse_bool(value):
    return (value == "True" or value == "true" or value == "1")

class CustomConfig(Config):
    device_id: str = None
    data_path = "/opt/data"
    device_name: str = None
    logger_level = "debug"
    check_data_anomalies: bool = False
    check_data_extreme_outlier: bool = True
    check_data_schema: bool = True
    check_receive_time_outlier: bool = True

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        self.check_data_anomalies = parse_bool(self.check_data_anomalies)
        self.check_data_extreme_outlier = parse_bool(self.check_data_extreme_outlier)
        self.check_data_schema = parse_bool(self.check_data_schema)
        self.check_receive_time_outlier = parse_bool(self.check_receive_time_outlier)

class Operator(OperatorBase):
    configType = CustomConfig
    
    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)

        if not os.path.exists(self.config.data_path):
            os.mkdir(self.config.data_path)

        self.device_id = self.config.device_id

        self.device_name = self.config.device_name

        self.active = []

        self.init_phase_duration = pd.Timedelta(2,'d')
        self.setup_operator_start(self.config.data_path)
        self.first_data_time = None

        if self.config.check_data_schema:
            print("Data Schema Detector is active")

        if self.config.check_data_anomalies:
            print("Curve Explorer is active!")
            self.Curve_Explorer = curve_anomaly.Curve_Explorer(self.config.data_path)
            self.active.append(self.Curve_Explorer)
        
        if self.config.check_data_extreme_outlier:
            print("Point Explorer is active!")
            self.Point_Explorer = point_outlier.Point_Explorer(os.path.join(self.config.data_path, "point_explorer"))
            self.active.append(self.Point_Explorer)

        if self.config.check_receive_time_outlier:
            print("Frequency Monitor is active!")
            self.frequency_monitor = FrequencyDetector(
                kafka_produce_func=self.produce,
                data_path=os.path.join(self.config.data_path, "frequency_monitor")
            )
            self.frequency_monitor.start()

        self.Consumption_Explorer = consumption_anomaly.Consumption_Explorer(os.path.join(self.config.data_path, "consumption_explorer"))

        self.send_init_message()

    def setup_operator_start(self, data_path):
        self.operator_start_time = utils.load_operator_start_time(data_path)
        if not self.operator_start_time:
            self.operator_start_time = datetime.datetime.now()
            utils.save_operator_start_time(data_path, self.operator_start_time)

    def input_is_real_time(self, timestamp):
        return timestamp >= self.operator_start_time

    def handle_frequency_monitor(self, timestamp):
        if self.frequency_monitor and self.input_is_real_time(timestamp):
            self.frequency_monitor.register_input(timestamp)

            if timestamp-self.operator_start_time > self.init_phase_duration:
                self.frequency_monitor.start_loop()

    def generate_init_message(self, minutes_until_start=None):
        if not minutes_until_start:
            minutes_until_start = int(self.init_phase_duration.total_seconds()/60)

        return {
                "type": "",
                "sub_type": "",
                "unit": "",
                "value": "",
                "initial_phase": f"Die Anwendung befindet sich noch für ca. {minutes_until_start} Minuten in der Initialisierungsphase"
            }

    def send_init_message(self):
        self.produce(self.generate_init_message())

    def run(self, data, selector='energy_func'):
        # These operators will also run when historic data is consumed and the init phase is completed based on historic timestamps 
        timestamp = utils.todatetime(data['time']).tz_localize(None)
        print('Input time: '+str(timestamp))
        if self.first_data_time == None:
            self.first_data_time = timestamp

        self.handle_frequency_monitor(timestamp)

        for operator in self.active:
            # each operator has to check for 2days init phase
            sample_is_anomalous, result = operator.run(data)

            # only return when input is realtime, historic data is only used for training
            if sample_is_anomalous:
                print(f"Anomaly occured: Detector={result['type']} Value={result['value']}")
                if self.input_is_real_time(timestamp):
                    return result 

        # Check init phase
        # Use input timestamp and first input for historic and real time data support 
        if timestamp-self.first_data_time < self.init_phase_duration:
            print("Still in initialisation phase!")
            td_until_start = self.init_phase_duration - (timestamp - self.first_data_time)
            minutes_until_start = int(td_until_start.total_seconds()/60)
            return self.generate_init_message(minutes_until_start)
            

        self.Consumption_Explorer.run(data)

    def stop(self):
        super().stop()

        if self.config.check_receive_time_outlier:
            self.frequency_monitor.stop()
            self.frequency_monitor.save()

        if self.config.check_data_anomalies:
            self.Curve_Explorer.save()

        if self.config.check_data_extreme_outlier:
            self.Point_Explorer.save()
