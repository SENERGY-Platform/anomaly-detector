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

    def create_debug_result(self):
        import numpy as np
        import pandas as pd 
        def random_df():
            start = pd.to_datetime('2015-01-01')
            end = pd.to_datetime('2015-01-03')
            n = 100
            ts = random_dates(start, end, n)
            data = pd.DataFrame({"value": np.random.rand(n,), "reconstr": np.random.rand(n,)}, index=ts)
            return data
        
        def random_dates(start, end, n=10):
            start_u = start.value//10**9
            end_u = end.value//10**9

            return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

        df = random_df()
        return {
                    "type": "curve_anomaly",
                    "sub_type": "",
                    "message": "An anomaly occured",
                    "value": 30.89,
                    "original_reconstructed_curves": df.reset_index().to_json(orient="values"),
                    "start_time": df.index[0].isoformat(),
                    "end_time": df.index[-1].isoformat()
        }

    def save(self):
        super().save()
        cont_utils.save_data(self.filename_dict, self.last_training_time, self.training_performance, self.model, self.training_max, self.reconstruction_errors)

    def stop(self):
        self.save()

    def update_with_new_value(self, value, timestamp):
        pass