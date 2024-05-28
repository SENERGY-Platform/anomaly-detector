import os 
from algo import utils
import pandas as pd 

class CurveDetector():
    def __init__(self, data_path, device_type, init_median, first_data_time):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        self.filename_dict = {"data": f'{data_path}/data.parquet', "last_training_time": f'{data_path}/last_training_time.pickle',
                         "anomalies": f'{data_path}/anomalies.pickle', "training_performance": f'{data_path}/training_performance.pickle',
                         "loads": f'{data_path}/loads.pickle', "model": f'{data_path}/model.pt'}
        self.first_data_time = first_data_time
        self.device_type = device_type
        self.init_median = init_median

        (self.data_list, 
         self.last_training_time,
         self.anomalies, 
         self.training_performance, 
         self.loads, 
         self.model) = utils.load_data(self.filename_dict, 
                                       self.data_list,
                                       self.last_training_time,  
                                       self.anomalies, 
                                       self.training_performance, 
                                       self.loads, 
                                       self.model)

    def save(self):
        utils.save_data(self.filename_dict, self.last_training_time, self.data_list,
                              self.model, self.training_performance, self.anomalies, self.loads)

    def stop(self):
        self.save()