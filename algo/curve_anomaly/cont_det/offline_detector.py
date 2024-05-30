from algo.curve_anomaly.cont_det.cont_detector import ContCurveDetector
from algo import utils
from .. import cont_device
import pandas as pd 
import torch 


class OfflineTrainContCurveDetector(ContCurveDetector):
    # Used for training inside the operator
    def __init__(self, data_path, device_type, init_median, first_data_time):
        super().__init__(data_path, device_type, init_median, first_data_time)

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
        self.last_training_time, self.model, self.training_performance, self.training_max = self.batch_train(self.data_list, self.first_data_time, self.last_training_time, self.model, use_cuda, self.training_performance, self.training_max)
        test_result, self.loads, self.anomalies, self.reconstruction_errors = self.test(self.data_list, self.first_data_time, self.last_training_time, self.model, use_cuda, self.anomalies, self.reconstruction_errors, self.training_max)
        if test_result=='cont_device_anomaly':
            time_window_start = (timestamp-pd.Timedelta(1,'hour')).floor('min')
            self.timestamp_last_anomaly, anomaly_during_last_30_min = cont_device.notification_decision(
                                                                       self.timestamp_last_anomaly, timestamp)
            if anomaly_during_last_30_min:
                return True, self.create_result(f'In der Zeit seit {str(time_window_start)} wurde eine Anomalie im Lastprofil festgestellt.', str(time_window_start), "continous_device")
            else:
                return False, ''

    def batch_train(self, data_list, first_data_time, last_training_time, model, use_cuda, training_performance, training_max):
        current_timestamp = utils.todatetime(data_list[-1][0]).tz_localize(None)
        if current_timestamp-last_training_time.tz_localize(None) >= pd.Timedelta(14, 'days'): 
            if last_training_time.tz_localize(None) == first_data_time.tz_localize(None):
                model = cont_device.Autoencoder(32)
                if use_cuda:
                    model = model.cuda()
            model, training_performance, training_max = cont_device.batch_train(data_list, model, use_cuda, training_performance)
            
            last_training_time = current_timestamp
            return last_training_time, model, training_performance, training_max
        elif current_timestamp-last_training_time.tz_localize(None) < pd.Timedelta(14, 'days'):
            return last_training_time, model, training_performance, training_max

    def test(self, data_list, first_data_time, last_training_time, model, use_cuda, anomalies, reconstruction_errors, training_max):
        if last_training_time.tz_localize(None) > first_data_time.tz_localize(None):
            output, anomalies, reconstruction_errors = cont_device.test(data_list, model, use_cuda, anomalies, training_max, reconstruction_errors)
            return output, anomalies, reconstruction_errors
        else:
            return None, anomalies, reconstruction_errors

