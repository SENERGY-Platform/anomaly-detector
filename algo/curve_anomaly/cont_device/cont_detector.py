from algo.curve_anomaly.curve_detector import CurveDetector

import pandas as pd 

class ContCurveDetector(CurveDetector):
    # Used for device that send continously data
    def __init__(self, data_path, device_type, init_median, first_data_time):
        super().__init__(data_path, device_type, init_median, first_data_time)

    def create_result(self, message, value, sub_type):
        _, last_anomalous_tw_smooth, last_anomalous_tw_reconstructed = self.anomalies[-1] # First two objects are Series, last object is flattended np array
        df_smooth_and_reconstr = pd.DataFrame(last_anomalous_tw_smooth).assign(reconstr=last_anomalous_tw_reconstructed)
        return {
                    "type": "curve_anomaly",
                    "sub_type": sub_type,
                    "message": message,
                    "value": value,
                    "original_reconstructed_curves": df_smooth_and_reconstr.reset_index(inplace=True).to_json(orient="values")
        }