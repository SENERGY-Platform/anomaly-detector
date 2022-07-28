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
import torch
import pandas as pd
import os
import pickle
from . import anom_detector, cont_device, load_device

class Operator(util.OperatorBase):
    def __init__(self, device_id, data_path):
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        self.device_id = device_id

        self.model_file_path = f'{data_path}/{self.device_id}_model.pt'
        self.anomaly_detector_path = f'{data_path}/{self.device_id}_anomaly_detector.pickle'

        self.anomaly_detector = anom_detector.Anomaly_Detector(device_id)

    def get_device_type(self,data_list):# entries in data_list are of the form (timestamp, data point)
        data_series = pd.Series(data=[data_point for _, data_point in data_list], index=[timestamp for timestamp, _ in data_list]).sort_index()
        data_series = data_series[~data_series.index.duplicated(keep='first')]
        device_type = 'cont_device'
        for timestamp_1 in data_series.index:
            constantly_zero = True
            if timestamp_1 + pd.Timedelta(2,'hours') < data_series.index.max():
                for timestamp_2 in data_series.loc[timestamp_1:timestamp_1+pd.Timedelta(2,'hours')].index:
                    if data_series.loc[timestamp_2] != 0:
                        constantly_zero = False
                        break
                if constantly_zero == True:
                    device_type = 'load_device'
                    break    
        return device_type
        
    def batch_train(self, data, use_cuda):
        if pd.to_datetime(data['energy_time'], unit='ms')-self.anomaly_detector.last_training_time >= pd.Timedelta(14, 'days'): 
            if self.anomaly_detector.device_type == 'cont_device':
                if self.anomaly_detector.last_training_time == self.anomaly_detector.initial_time:
                    self.anomaly_detector.model = cont_device.Autoencoder(32)
                    if use_cuda:
                        self.anomaly_detector.model = self.anomaly_detector.model.cuda()
                self.anomaly_detector.model = cont_device.batch_train(self.anomaly_detector, self.model_file_path, use_cuda)
            elif self.anomaly_detector.device_type == 'load_device':
                pass # training IsolationForest is that fast, that we can train it again with every new data point.
            self.anomaly_detector.last_training_time = pd.to_datetime(data['energy_time'], unit='ms')
        elif pd.to_datetime(data['energy_time'], unit='ms')-self.anomaly_detector.last_training_time < pd.Timedelta(14, 'days'):
            pass

    def test(self, use_cuda):
        if self.anomaly_detector.last_training_time > self.anomaly_detector.initial_time:
            if self.anomaly_detector.device_type == 'cont_device':
                output = cont_device.test(self.anomaly_detector.data, self.anomaly_detector, use_cuda)
            elif self.anomaly_detector.device_type == 'load_device':
                output = load_device.train_test(self.anomaly_detector, self.model_file_path)
            return output

    def run(self, data, selector='energy_func'):
        if os.getenv("DEBUG") is not None and os.getenv("DEBUG").lower() == "true":
            print(selector + ": " + 'energy: '+str(data['energy'])+'  '+'time: '+str(pd.to_datetime(data['energy_time'], unit='ms')))
        self.anomaly_detector.data.append([pd.to_datetime(data['energy_time'], unit='ms'), float(data['energy'])])
        if self.anomaly_detector.first_data_time == None:
            self.anomaly_detector.first_data_time = pd.to_datetime(data['energy_time'], unit='ms')
            self.anomaly_detector.last_training_time = self.anomaly_detector.first_data_time
        if pd.to_datetime(data['energy_time'], unit='ms') < self.anomaly_detector.initial_time:
            return
        if self.anomaly_detector.device_type == None:
            if pd.to_datetime(data['energy_time'], unit='ms')-self.anomaly_detector.first_data_time < pd.Timedelta(1, 'days'):
                return
            elif pd.to_datetime(data['energy_time'], unit='ms')-self.anomaly_detector.first_data_time >= pd.Timedelta(1, 'days'):
                self.anomaly_detector.device_type = self.get_device_type(self.anomaly_detector.data)
                print(self.anomaly_detector.device_type)
        use_cuda = torch.cuda.is_available()
        self.batch_train(data, use_cuda)
        output = self.test(use_cuda)
        with open(self.anomaly_detector_path, 'wb') as f:
            pickle.dump(self.anomaly_detector)
        return output
