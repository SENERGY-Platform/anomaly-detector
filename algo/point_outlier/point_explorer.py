import pandas as pd
from algo import utils
import operator_lib.util as util


__all__ = ("Point_Explorer",)
LOG_PREFIX = "POINT_DETECTOR"

class Point_Explorer(utils.StdPointOutlierDetector):
    def __init__(self, data_path, device_type, init_median):
        super().__init__(data_path)
        self.active = False # Introduce this variable to constantly check
        self.device_type = device_type
        self.init_median = init_median

    def check(self, value, timestamp):
        new_value = float(value)
            
        anomaly_occured = False
        threshold = None
        if self.point_is_anomalous_high(new_value):
            sub_type = "high"
            anomaly_occured = True
            threshold = self.get_upper_threshold()

        if self.point_is_anomalous_low(new_value):
            sub_type = "low"
            anomaly_occured = True
            threshold = self.get_lower_threshold()
                
        if anomaly_occured:
            util.logger.info(f'{LOG_PREFIX}: An extreme point outlier just occured!')
            return True, {
                    "type": "extreme_value",
                    "sub_type": sub_type,
                    "value": new_value,
                    "threshold": round(threshold, 2),
                    "mean": round(self.current_mean, 2)
            }
        
        return False, {}
    
    def update_with_new_value(self, value, timestamp): 
        if self.device_type == "load_device":
            if not self.active and value > self.init_median + 10:
                self.active = True
                start_of_end = None
            elif self.active:
                if value <= self.init_median + 1 and not start_of_end:
                    start_of_end = timestamp
                elif value <= self.init_median + 1 and start_of_end:
                    if timestamp - start_of_end >= pd.Timedelta(10,"min"):
                        self.active = False
                elif value > self.init_median + 1 and start_of_end:
                    start_of_end = None
                
        if self.active or self.device_type == "cont_device":
            self.update(value)
            self.save()
            
