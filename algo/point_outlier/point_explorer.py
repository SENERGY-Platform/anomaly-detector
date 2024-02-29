import pandas as pd
from algo import utils


__all__ = ("Point_Explorer",)
LOG_PREFIX = "POINT_DETECTOR"

class Point_Explorer(utils.StdPointOutlierDetector):
    def __init__(self, data_path):
        super().__init__(data_path)

    def check(self, data):
        new_value = float(data['value'])
            
        anomaly_occured = False
        if self.point_is_anomalous_high(new_value):
            sub_type = "high"
            anomaly_occured = True

        if self.point_is_anomalous_low(new_value):
            sub_type = "low"
            anomaly_occured = True
                
        if anomaly_occured:
            print(f'{LOG_PREFIX}: An extreme point outlier just occured! \n\n\n\n')
            return True, {
                    "type": "extreme_value",
                    "sub_type": sub_type,
                    "value": new_value,
                    "unit": "TODO"
            }
        
        return False, {}

    def update(self, value):         
        self.update(value)
        self.save()
            
