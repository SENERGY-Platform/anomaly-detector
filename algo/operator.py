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

import util
import os
from algo import curve_anomaly
from algo import point_outlier

class Operator(util.OperatorBase):
    def __init__(
        self, 
        device_id, 
        data_path, 
        device_name='Gerät',
        check_data_extreme_outlier=True,
        check_data_anomalies=True,
        check_data_schema=True,
        frequency_monitor=None
    ):
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        self.device_id = device_id

        self.device_name = device_name

        self.active = []
        self.frequency_monitor = frequency_monitor

        if check_data_anomalies:
            print("Curve Explorer is active!")
            self.Curve_Explorer = curve_anomaly.Curve_Explorer(data_path)
            self.active.append(self.Curve_Explorer)
        
        if check_data_extreme_outlier:
            print("Point Explorer is active!")
            self.Point_Explorer = point_outlier.Point_Explorer()
            self.active.append(self.Point_Explorer)

        if frequency_monitor:
            print("Frequency Monitoring is active!")
            self.frequency_monitor = frequency_monitor

    def run(self, data, selector='energy_func'):
        for operator in self.active:
            if self.frequency_monitor:
                self.frequency_monitor.register_input(data)
            sample_is_anomalous, message = operator.run(data)

            if sample_is_anomalous:
                return True, message
