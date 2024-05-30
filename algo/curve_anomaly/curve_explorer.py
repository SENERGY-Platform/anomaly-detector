from . import curve_utils, cont_device
from algo import utils
import pandas as pd
import torch
import os


__all__ = ("Curve_Explorer",)
LOG_PREFIX = "CURVE_DETECTOR"

class Curve_Explorer:
    def __init__(self, data_path, device_type, init_median, first_data_time):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        self.filename_dict = {"data": f'{data_path}/data.parquet', "last_training_time": f'{data_path}/last_training_time.pickle',
                         "anomalies": f'{data_path}/anomalies.pickle', "training_performance": f'{data_path}/training_performance.pickle',
                         "loads": f'{data_path}/loads.pickle', "model": f'{data_path}/model.pt',
                         "training_max": f'{data_path}/training_max.pickle', "reconstruction_errors": f'{data_path}/reconstruction_errors.pickle'}

        self.first_data_time = first_data_time
        self.last_training_time = self.first_data_time
        self.timestamp_last_anomaly = pd.Timestamp.min
        self.data_list = []
        self.model = None
        self.training_performance = []
        self.anomalies = []
        self.device_type = device_type
        self.loads = []
        self.init_median = init_median
        self.training_max = None
        self.reconstruction_errors = None

        (self.data_list, 
         self.last_training_time,
         self.anomalies, 
         self.training_performance, 
         self.loads, 
         self.model,
         self.training_max,
         self.reconstruction_errors) = utils.load_data(self.filename_dict, 
                                       self.data_list,
                                       self.last_training_time,  
                                       self.anomalies, 
                                       self.training_performance, 
                                       self.loads, 
                                       self.model,
                                       self.training_max,
                                       self.reconstruction_errors)




    def check(self, value, timestamp):
        if self.first_data_time == None:
            self.first_data_time = timestamp
            self.last_training_time = self.first_data_time
            self.data_list.append([timestamp, value])
            return False, ''
        self.data_list.append([timestamp, value])
        if self.data_list[-1][0] - self.data_list[0][0] >= pd.Timedelta(50, "d"): # Only keep data, which is at most 50 days old.
            del self.data_list[0]
        use_cuda = torch.cuda.is_available()
        self.last_training_time, self.model, self.training_performance, self.training_max = curve_utils.batch_train(self.data_list, self.first_data_time, self.last_training_time, self.device_type, self.model, use_cuda, self.training_performance, self.training_max)
        test_result, self.loads, self.anomalies, self.reconstruction_errors = curve_utils.test(self.data_list, self.first_data_time, self.last_training_time, self.device_type, self.model, use_cuda, self.anomalies, self.loads, self.init_median, self.reconstruction_errors, self.training_max)
        if test_result=='cont_device_anomaly':
            time_window_start = (timestamp-pd.Timedelta(1,'hour')).floor('min')
            self.timestamp_last_anomaly, anomaly_during_last_30_min = cont_device.notification_decision(
                                                                       self.timestamp_last_anomaly, timestamp)
            if anomaly_during_last_30_min:
                return True, self.create_result(f'In der Zeit seit {str(time_window_start)} wurde eine Anomalie im Lastprofil festgestellt.', str(time_window_start), "continous_device")
            else:
                return False, ''
        elif test_result=='load_device_anomaly_power_curve':
            return True, self.create_result(f'Bei der letzten Benutzung wurde eine Anomalie im Lastprofil festgestellt.', "", "uncontinious_device_curve")
        elif test_result=='load_device_anomaly_length':
            return True, self.create_result(f'Bei der letzten Benutzung wurde eine ungewöhnliche Laufdauer festgestellt.', "", "uncontinious_device_length")
        else:
            return False, ''

    def update_with_new_value(self, data, ts):
        pass

    def create_result(self, message, value, sub_type):
        if self.device_type=="cont_device":
            last_anomalous_tw, last_anomalous_tw_smooth, last_anomalous_tw_reconstructed = self.anomalies[-1] # First two objects are Series, last object is flattended np array
            df_smooth_and_reconstr = pd.DataFrame(last_anomalous_tw_smooth).assign(reconstr=last_anomalous_tw_reconstructed)
            return {
                    "type": "curve_anomaly",
                    "sub_type": sub_type,
                    "message": message,
                    "value": value,
                    "original_reconstructed_curves": df_smooth_and_reconstr.reset_index(inplace=True).to_json(orient="values")
            }
        else:
            return {
                    "type": "curve_anomaly",
                    "sub_type": sub_type,
                    "message": message,
                    "value": value
            }

    def save(self):
        utils.save_data(self.filename_dict, self.last_training_time, self.data_list,
                              self.model, self.training_performance, self.anomalies, self.loads, self.training_max, self.reconstruction_errors)

    def stop(self):
        self.save()