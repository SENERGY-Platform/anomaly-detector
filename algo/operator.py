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
import pyarrow
from . import anom_detector, cont_device, load_device


class Operator(util.OperatorBase):
    def __init__(self, device_id, data_path, device_name='Gerät'):
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        self.device_id = device_id

        self.device_name = device_name

        self.anomaly_in_last_datapoint = False


        self.model_file_path = f'{data_path}/{self.device_id}_model.pt'
        self.anomaly_detector_data_path = f'{data_path}/{self.device_id}_anomaly_detector_data.parquet'
        self.anomaly_detector_initial_time_path = f'{data_path}/{self.device_id}_anomaly_detector_initial_time.pickle'
        self.anomaly_detector_first_data_time_path = f'{data_path}/{self.device_id}_anomaly_detector_first_data_time.pickle'
        self.anomaly_detector_last_training_time_path = f'{data_path}/{self.device_id}_anomaly_detector_last_training_time.pickle'
        self.anomaly_detector_device_id_path = f'{data_path}/{self.device_id}_anomaly_detector_device_id.pickle'
        self.anomaly_detector_device_type_path = f'{data_path}/{self.device_id}_anomaly_detector_device_type.pickle'
        self.anomaly_detector_anomalies_path = f'{data_path}/{self.device_id}_anomaly_detector_anomalies.pickle'
        self.anomaly_detector_training_performance_path = f'{data_path}/{self.device_id}_anomaly_detector_training_performance.pickle'
        self.anomaly_detector_loads_path = f'{data_path}/{self.device_id}_anomaly_detector_loads.pickle'

        self.anomaly_detector = anom_detector.Anomaly_Detector(device_id)

        '''if os.path.exists(self.anomaly_detector_data_path):
            df = pd.read_parquet(self.anomaly_detector_data_path)
            df.index = pd.to_datetime(df.index)
            data_series = pd.Series(data=df['power_values'], index=df.index)
            data_series = df[~df.index.duplicated(keep='first')]
            for i in range(len(data_series.index)):
                self.anomaly_detector.data.append([self.todatetime(data_series.index[i]), float(data_series.iloc[i])])
            print(len(data_series.index))
            print('Old data loaded.')
        

        if os.path.exists(self.anomaly_detector_device_type_path):
            with open(self.anomaly_detector_device_type_path, 'rb') as f:
                self.anomaly_detector.device_type = pickle.load(f)
            print(self.anomaly_detector.device_type)
            if self.anomaly_detector.device_type == 'cont_device':
                if os.path.exists(self.model_file_path):
                    self.anomaly_detector.model = cont_device.Autoencoder(32)
                    self.anomaly_detector.model.load_state_dict(torch.load(self.model_file_path))
                    print('Pretrained model loaded!')
            elif self.anomaly_detector.device_type == 'load_device':
                if os.path.exists(self.anomaly_detector_loads_path):
                    with open(self.anomaly_detector_loads_path, 'rb') as f:
                        self.anomaly_detector.loads = pickle.load(f)
                    print('List of loads from the past loaded!')'''
                    

    def todatetime(self, timestamp):
        if str(timestamp).isdigit():
            if len(str(timestamp))==13:
                return pd.to_datetime(int(timestamp), unit='ms')
            elif len(str(timestamp))==19:
                return pd.to_datetime(int(timestamp), unit='ns')
        else:
            return pd.to_datetime(timestamp)
    
    def get_device_type(self,data_list):# entries in data_list are of the form (timestamp, data point)
        data_series = pd.Series(data=[data_point for _, data_point in data_list], index=[timestamp for timestamp, _ in data_list]).sort_index()
        data_series = data_series[~data_series.index.duplicated(keep='first')]
        device_type = 'cont_device'
        for timestamp_1 in data_series.index:
            constantly_zero = True
            if timestamp_1 + pd.Timedelta(2,'hours') < data_series.index.max():
                for timestamp_2 in data_series.loc[timestamp_1:timestamp_1+pd.Timedelta(2,'hours')].index:
                    if data_series.loc[timestamp_2] > 5:
                        constantly_zero = False
                        break
                if constantly_zero == True:
                    device_type = 'load_device'
                    break    
        return device_type
        
    def batch_train(self, data, use_cuda):
        if self.todatetime(data['energy_time']).tz_localize(None)-self.anomaly_detector.last_training_time >= pd.Timedelta(14, 'days'): 
            if self.anomaly_detector.device_type == 'cont_device':
                if self.anomaly_detector.last_training_time == self.anomaly_detector.first_data_time:
                    self.anomaly_detector.model = cont_device.Autoencoder(32)
                    if use_cuda:
                        self.anomaly_detector.model = self.anomaly_detector.model.cuda()
                self.anomaly_detector.model = cont_device.batch_train(self.anomaly_detector, self.model_file_path, use_cuda)
            elif self.anomaly_detector.device_type == 'load_device':
                pass # training IsolationForest is that fast, that we can train it again with every new data point.
            self.anomaly_detector.last_training_time = self.todatetime(data['energy_time']).tz_localize(None)
        elif self.todatetime(data['energy_time']).tz_localize(None)-self.anomaly_detector.last_training_time < pd.Timedelta(14, 'days'):
            pass

    def test(self, use_cuda):
        if self.anomaly_detector.device_type == 'cont_device' and self.anomaly_detector.last_training_time > self.anomaly_detector.first_data_time:
            output = cont_device.test(self.anomaly_detector.data, self.anomaly_detector, use_cuda)
            return output
        elif self.anomaly_detector.device_type == 'load_device':
            output = load_device.train_test(self.anomaly_detector, self.model_file_path)
            return output
        else:
            return 

    def save_data(self):
        data_list = self.anomaly_detector.data
        data_series = pd.Series(data=[data_point for _, data_point in data_list], index=[timestamp.replace(microsecond=0).strftime('%Y-%m-%d %X') for timestamp, _ in data_list]).sort_index()
        data_series = data_series[~data_series.index.duplicated(keep='first')]
        df = data_series.to_frame()
        df.columns = ['power_values']
        df.to_parquet(self.anomaly_detector_data_path)
        with open(self.anomaly_detector_initial_time_path, 'wb') as f:
            pickle.dump(self.anomaly_detector.initial_time, f)
        with open(self.anomaly_detector_first_data_time_path, 'wb') as f:
            pickle.dump(self.anomaly_detector.first_data_time, f)
        with open(self.anomaly_detector_last_training_time_path, 'wb') as f:
            pickle.dump(self.anomaly_detector.last_training_time, f)
        with open(self.anomaly_detector_device_id_path, 'wb') as f:
            pickle.dump(self.anomaly_detector.device_id, f)
        with open(self.anomaly_detector_device_type_path, 'wb') as f:
            pickle.dump(self.anomaly_detector.device_type, f)
        with open(self.anomaly_detector_anomalies_path, 'wb') as f:
            pickle.dump(self.anomaly_detector.anomalies, f)
        with open(self.anomaly_detector_training_performance_path, 'wb') as f:
            pickle.dump(self.anomaly_detector.training_performance, f)
        with open(self.anomaly_detector_loads_path, 'wb') as f:
            pickle.dump(self.anomaly_detector.loads, f)
    
    def run(self, data, selector='energy_func'):
        timestamp = self.todatetime(data['energy_time']).tz_localize(None)
        print('energy: '+str(data['energy'])+'  '+'time: '+str(timestamp))
        if self.anomaly_detector.first_data_time == None:
            self.anomaly_detector.first_data_time = timestamp
            self.anomaly_detector.last_training_time = self.anomaly_detector.first_data_time
            self.anomaly_detector.data.append([timestamp, float(data['energy'])])
            return
        if self.anomaly_detector.device_type == None:
            if timestamp-self.anomaly_detector.first_data_time < pd.Timedelta(1, 'days'):
                return
            elif timestamp-self.anomaly_detector.first_data_time >= pd.Timedelta(1, 'days'):
                self.anomaly_detector.device_type = self.get_device_type(self.anomaly_detector.data)
                print(self.anomaly_detector.device_type)
        self.anomaly_detector.data.append([timestamp, float(data['energy'])])
        use_cuda = torch.cuda.is_available()
        self.batch_train(data, use_cuda)
        test_result = self.test(use_cuda)
        self.save_data()
        if test_result=='cont_device_anomaly':
            if self.anomaly_in_last_datapoint==False:
                time_window_start = (timestamp-pd.Timedelta(1,'hour')).floor('min')
                self.anomaly_in_last_datapoint=True
                return {'anomaly': f'Nachricht vom {str(timestamp.date())} um {str(timestamp.hour)}:{str(timestamp.minute)} Uhr: In der Zeit seit {str(time_window_start)} wurde beim {self.device_name} eine Anomalie im Lastprofil festgestellt.'}
            elif self.anomaly_in_last_datapoint==True:
                return
        elif test_result=='load_device_anomaly_power_curve':
            return {'anomaly':f'Nachricht vom {str(timestamp.date())} um {str(timestamp.hour)}:{str(timestamp.minute)} Uhr: Bei der letzten Benutzung vom {self.device_name} wurde eine Anomalie im Lastprofil festgestellt.'}
        elif test_result=='load_device_anomaly_length':
            return {'anomaly':f'Nachricht vom {str(timestamp.date())} um {str(timestamp.hour)}:{str(timestamp.minute)} Uhr: Bei der letzten Benutzung vom {self.device_name} wurde eine ungewöhnliche Laufdauer festgestellt.'}
        else:
            self.anomaly_in_last_datapoint=False
            return
