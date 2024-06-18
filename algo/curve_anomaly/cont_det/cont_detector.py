from algo.curve_anomaly.curve_detector import CurveDetector

import pandas as pd 
from . import cont_utils

class ContCurveDetector(CurveDetector):
    # Used for device that send continously data
    def __init__(self, data_path, init_median, first_data_time):
        super().__init__(data_path, init_median, first_data_time)

        self.filename_dict.update({"last_training_time": f'{data_path}/last_training_time.pickle',
                              "training_performance": f'{data_path}/training_performance.pickle',
                              "model": f'{data_path}/model.pt',
                              "training_max": f'{data_path}/training_max.pickle',
                              "reconstruction_errors": f'{data_path}/reconstruction_errors.pickle'})

        self.last_training_time = self.first_data_time
        self.timestamp_last_anomaly = pd.Timestamp.min
        
        self.training_performance = []
        self.model = None
        self.training_max = None
        self.reconstruction_errors = None
        self.data_list = []

        (self.last_training_time,
         self.training_performance,  
         self.model,
         self.training_max,
         self.reconstruction_errors) = cont_utils.load_data(self.filename_dict,
                                       self.last_training_time,  
                                       self.training_performance, 
                                       self.model,
                                       self.training_max,
                                       self.reconstruction_errors)

    def create_result(self, message, value, sub_type):
        _, last_anomalous_tw_smooth, last_anomalous_tw_reconstructed = self.anomalies[-1] # First two objects are Series, last object is flattended np array
        df_smooth_and_reconstr = pd.DataFrame(last_anomalous_tw_smooth).assign(reconstr=last_anomalous_tw_reconstructed)
        return {
                    "type": "curve_anomaly",
                    "sub_type": sub_type,
                    "message": message,
                    "value": value,
                    "original_reconstructed_curves": df_smooth_and_reconstr.reset_index().to_json(orient="values"),
                    "start_time": df_smooth_and_reconstr.index[0].isoformat(),
                    "end_time": df_smooth_and_reconstr.index[-1].isoformat()
        }
    
    def save(self):
        super().save()
        cont_utils.save_data(self.filename_dict, self.last_training_time, self.training_performance, self.model, self.training_max, self.reconstruction_errors)

    def stop(self):
        self.save()

    def update_with_new_value(self, value, timestamp):
        pass