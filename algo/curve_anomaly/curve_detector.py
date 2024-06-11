import os 
from . import curve_utils
 

class CurveDetector():
    def __init__(self, data_path, init_median, first_data_time):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        self.filename_dict = {"data": f'{data_path}/data.parquet', "anomalies": f'{data_path}/anomalies.pickle'}
        self.first_data_time = first_data_time
        self.init_median = init_median
        self.data_path = data_path
        
        self.data_list = []
        self.anomalies = []

        (self.data_list, self.anomalies) = curve_utils.load_data(self.filename_dict, self.data_list, self.anomalies)

    def save(self):
        curve_utils.save_data(self.filename_dict, self.data_list, self.anomalies)

    def stop(self):
        self.save()