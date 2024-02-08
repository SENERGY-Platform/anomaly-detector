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

from algo import curve_anomaly
from algo import point_outlier
from algo import consumption_anomaly
from algo import utils
import util
import pandas as pd
import datetime

class Operator(util.OperatorBase):
    def __init__(
        self, 
        device_id, 
        data_path, 
        device_name='Gerät',
        check_data_extreme_outlier=True,
        check_data_anomalies=True,
        check_data_schema=True,
        frequency_monitor=None,
    ):
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        self.device_id = device_id

        self.device_name = device_name

        self.active = []
        self.frequency_monitor = frequency_monitor

        self.init_phase_duration = pd.Timedelta(2,'d')
        self.operator_start_time = datetime.datetime.now()
        self.first_data_time = None

        self.check_data_schema = check_data_schema
        if self.check_data_schema:
            print("Data Schema Detector is active")

        if check_data_anomalies:
            print("Curve Explorer is active!")
            self.Curve_Explorer = curve_anomaly.Curve_Explorer(data_path)
            self.active.append(self.Curve_Explorer)
        
        if check_data_extreme_outlier:
            print("Point Explorer is active!")
            self.Point_Explorer = point_outlier.Point_Explorer(os.path.join(data_path, "point_explorer"))
            self.active.append(self.Point_Explorer)

        if frequency_monitor:
            self.frequency_monitor = frequency_monitor

        self.Consumption_Explorer = consumption_anomaly.Consumption_Explorer(os.path.join(data_path, "consumption_explorer"))

    def input_is_real_time(self, timestamp):
        return timestamp >= self.operator_start_time

    def handle_frequency_monitor(self, timestamp):
        if self.frequency_monitor and self.input_is_real_time(timestamp):
            self.frequency_monitor.register_input(timestamp)

            if timestamp-self.first_data_time > self.init_phase_duration:
                self.frequency_monitor.start_loop()

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
            return {
                "type": "",
                "sub_type": "",
                "unit": "",
                "value": "",
                "initial_phase": f"Die Anwendung befindet sich noch für ca. {minutes_until_start} Minuten in der Initialisierungsphase"
            }
            

        self.Consumption_Explorer.run(data)
