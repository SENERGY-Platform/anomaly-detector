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
    def __init__(self, device_id, data_path, device_name='Gerät'):
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        self.device_id = device_id

        self.device_name = device_name

        self.Curve_Explorer = curve_anomaly.Curve_Explorer(data_path)
        self.Point_Explorer = point_outlier.Point_Explorer()
    
    def run(self, data, selector='energy_func'):
        self.Curve_Explorer.run(data)
        self.Point_Explorer.run(data)