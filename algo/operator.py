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
from . import anom_detector, cont_device, load_device
import pandas as pd
import os

class Operator(util.OperatorBase):
    def __init__(self, device_id, data_path):
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        self.model_file_path = f'{data_path}/{self.device_id}_model.pt'

        self.anomaly_detector = anom_detector.Anomaly_Detector(device_id)

    def get_device_type(data_list):# entris in data_list are of the form (timestamp, data point)
        data_series = pd.Series(data=[data_point for _, data_point in data_list], index=[timestamp for timestamp, _ in data_list])
        device_type = 'cont_device'
        for timestamp_1 in data_series.index:
            constantly_zero = True
            for timestamp_2 in data_series.loc[timestamp_1:timestamp_1+pd.Timedelta(2,'hours')]:
                if data_series.loc[timestamp_2] != 0:
                    constantly_zero = False
                    break
            if constantly_zero == True:
                device_type = 'load_device'
                break    
        return device_type
        
    def batch_train(self, data):
        if data['energy_time']-self.anomaly_detector.last_training_time >= pd.Timedelta(14, 'days'): 
            if self.anomaly_detector.device_type == 'cont_device':
                if self.anomaly_detector.last_training_time == self.anomaly_detector.initial_time:
                    self.anomaly_detector.model = cont_device.Autoencoder(32)
                self.anomaly_detector.model = cont_device.batch_train(self.anomaly_detector.model, self.anomaly_detector.data, self.model_file_path)
            elif self.anomaly_detector.device_type == 'load_device':
                pass # training IsolationForest is that fast, that we can train it again with every new data point.
            self.anomaly_detector.last_training_time = data['energy_time']
        elif data['energy_time']-self.anomaly_detector.last_training_time < pd.Timedelta(14, 'days'):
            pass

    def test(self):
        if self.anomaly_detector.last_training_time > self.anomaly_detector.initial_time:
            if self.anomaly_detector.device_type == 'cont_device':
                cont_device.test(self.anomaly_detector.data, self.anomaly_detector.model)
            elif self.anomaly_detector.device_type == 'load_device':
                self.anomaly_detector.model = load_device.train_test(self.anomaly_detector.data, self.model_file_path)
            return output

    def run(self, data, selector='energy_func'):
        if os.getenv("DEBUG") is not None and os.getenv("DEBUG").lower() == "true":
            print(selector + ": " + str(data))
        self.anomaly_detector.data.append([data['energy_time'], data['energy']])
        if self.anomaly_detector.first_data_time == None:
            self.anomaly_detector.first_data_time = data['energy_time']
        if data['energy_time'] < self.anomaly_detector.initial_time:
            return
        if self.anomaly_detector.device_type == None:
            if data['energy_time']-self.anomaly_detector.first_data_time < pd.Timedelta(1, 'days'):
                return
            elif data['energy_time']-self.anomaly_detector.first_data_time >= pd.Timedelta(1, 'days'):
                self.anomaly_detector.device_type = self.get_device_type(self.anomaly_detector.data)
        self.batch_train(data)
        output = self.test()
        return output
