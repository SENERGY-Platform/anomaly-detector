from algo.curve_anomaly.curve_detector import CurveDetector
from . import load_utils

class LoadCurveDetector(CurveDetector):
    def __init__(self, data_path, init_median, first_data_time):
        super().__init__(data_path, init_median, first_data_time)

        self.filename_dict.update({"loads": f'{data_path}/loads.pickle', "endpoint_last_load": f'{data_path}/endpoint_last_load.pickle'})
        
        self.loads, self.endpoint_last_load = load_utils.load_data(self.filename_dict)
        if not self.endpoint_last_load:
            self.endpoint_last_load = self.first_data_time

    def check(self, value, timestamp):
        if self.first_data_time == None:
            self.first_data_time = timestamp
            self.data_list.append([timestamp, value])
            return False, ''
        self.data_list.append([timestamp, value])
        i = 0
        while self.data_list[i][0] < self.endpoint_last_load: # Only keep data, which was sent since the last load ended.
            del self.data_list[0]
            i += 1
        test_result, self.loads, self.anomalies, self.endpoint_last_load = self.test(self.data_list, self.anomalies, self.loads, self.init_median, self.endpoint_last_load)

        if test_result=='load_device_anomaly_power_curve':
            return True, self.create_result(f'Bei der letzten Benutzung wurde eine Anomalie im Lastprofil festgestellt.', "", "uncontinious_device_curve")
        elif test_result=='load_device_anomaly_length':
            return True, self.create_result(f'Bei der letzten Benutzung wurde eine ungewöhnliche Laufdauer festgestellt.', "", "uncontinious_device_length")
        else:
            return False, ''
    
    def test(self, data_list, anomalies, loads, init_median):
        output, loads, anomalies, self.endpoint_last_load = load_utils.train_test(data_list, loads, anomalies, init_median)
        return output,  loads, anomalies, self.endpoint_last_load

    def create_result(self, message, value, sub_type):
        return {
                    "type": "curve_anomaly",
                    "sub_type": sub_type,
                    "message": message,
                    "value": value
        }
    
    def save(self):
        super().save()
        load_utils.save_data(self.filename_dict, self.loads, self.endpoint_last_load)

    def stop(self):
        self.save()