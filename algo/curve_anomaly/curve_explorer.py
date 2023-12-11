from . import curve_utils, cont_device
from algo import utils
import pandas as pd
import torch

__all__ = ("Curve_Explorer",)
    
class Curve_Explorer:
    def __init__(self, data_path):
        self.filename_dict = {"data": f'{data_path}/data.parquet', "initial_time": f'{data_path}/initial_time.pickle', "first_data_time": f'{data_path}/first_data_time.pickle',
                         "last_training_time": f'{data_path}/last_training_time.pickle', "device_type": f'{data_path}/device_type.pickle',
                         "anomalies": f'{data_path}/anomalies.pickle', "training_performance": f'{data_path}/training_performance.pickle',
                         "loads": f'{data_path}/loads.pickle', "model": f'{data_path}/model.pt'}

        self.initial_time = pd.Timestamp.now()
        self.first_data_time = None
        self.last_training_time = None
        self.timestamp_last_anomaly = pd.Timestamp.min
        self.timestamp_last_notification = pd.Timestamp.min
        self.data_list = []
        self.model = None
        self.training_performance = []
        self.anomalies = []
        self.device_type = None
        self.loads = []

        (self.data_list, 
         self.initial_time, 
         self.first_data_time, 
         self.last_training_time, 
         self.device_type, 
         self.anomalies, 
         self.training_performance, 
         self.loads, 
         self.model) = utils.load_data(self.filename_dict, 
                                       self.data_list, 
                                       self.initial_time, 
                                       self.first_data_time, 
                                       self.last_training_time, 
                                       self.device_type, 
                                       self.anomalies, 
                                       self.training_performance, 
                                       self.loads, 
                                       self.model)




    def run(self, data):
        timestamp = utils.todatetime(data['energy_time']).tz_localize(None)
        if self.initial_time-timestamp > pd.Timedelta(250,'days'):
            return
        print('energy: '+str(data['energy'])+'  '+'time: '+str(timestamp))
        if self.first_data_time == None:
            self.first_data_time = timestamp
            self.last_training_time = self.first_data_time
            self.data_list.append([timestamp, float(data['energy'])])
            return
        if self.device_type == None:
            if timestamp-self.first_data_time < pd.Timedelta(1, 'days'):
                self.data_list.append([timestamp, float(data['energy'])])
                return
            elif timestamp-self.first_data_time >= pd.Timedelta(1, 'days'):
                self.device_type = curve_utils.get_device_type(self.data_list)
                print(self.device_type)
        self.data_list.append([timestamp, float(data['energy'])])
        use_cuda = torch.cuda.is_available()
        self.last_training_time, self.model, self.training_performance = curve_utils.batch_train(self.data_list, self.first_data_time, self.last_training_time, self.device_type, self.model, use_cuda, self.training_performance)
        test_result, self.loads, self.anomalies = curve_utils.test(self.data_list, self.first_data_time, self.last_training_time, self.device_type, self.model, use_cuda, self.anomalies, self.loads)
        if test_result=='cont_device_anomaly':
            time_window_start = (timestamp-pd.Timedelta(1,'hour')).floor('min')
            self.timestamp_last_anomaly, self.timestamp_last_notification, notification_now = cont_device.notification_decision(
                                                                       self.timestamp_last_anomaly, self.timestamp_last_notification, timestamp)
            if notification_now:
                return {'anomaly': f'In der Zeit seit {str(time_window_start)} wurde eine Anomalie im Lastprofil festgestellt.'}
            else:
                return
        elif test_result=='load_device_anomaly_power_curve':
            return {'anomaly':f'Bei der letzten Benutzung wurde eine Anomalie im Lastprofil festgestellt.'}
        elif test_result=='load_device_anomaly_length':
            return {'anomaly':f'Bei der letzten Benutzung wurde eine ungewöhnliche Laufdauer festgestellt.'}
        

    def save(self):
        utils.save_data(self.filename_dict, self.initial_time, self.first_data_time, self.last_training_time, self.data_list,
                              self.model, self.training_performance, self.anomalies, self.device_type, self.loads)