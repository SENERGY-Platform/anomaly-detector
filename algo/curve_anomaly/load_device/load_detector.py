from algo.curve_anomaly.curve_detector import CurveDetector
import pandas as pd 
from algo.curve_anomaly import load_device

class LoadCurveDetector(CurveDetector):
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
        test_result, self.loads, self.anomalies, self.reconstruction_errors = self.test(self.data_list, self.anomalies, self.loads, self.init_median, self.reconstruction_errors)

        if test_result=='load_device_anomaly_power_curve':
            return True, self.create_result(f'Bei der letzten Benutzung wurde eine Anomalie im Lastprofil festgestellt.', "", "uncontinious_device_curve")
        elif test_result=='load_device_anomaly_length':
            return True, self.create_result(f'Bei der letzten Benutzung wurde eine ungewöhnliche Laufdauer festgestellt.', "", "uncontinious_device_length")
        else:
            return False, ''
    
    def test(self, data_list, anomalies, loads, init_median, reconstruction_errors):
        output, loads, anomalies = load_device.train_test(data_list, loads, anomalies, init_median)
        return output,  loads, anomalies, reconstruction_errors

    def create_result(self, message, value, sub_type):
        return {
                    "type": "curve_anomaly",
                    "sub_type": sub_type,
                    "message": message,
                    "value": value
        }