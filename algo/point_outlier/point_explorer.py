import pandas as pd
from algo import utils


__all__ = ("Point_Explorer",)

class Point_Explorer(utils.StdPointOutlierDetector):
    def __init__(self, data_path):
        super().__init__(data_path)

    def run(self, data):
        timestamp = utils.todatetime(data['energy_time']).tz_localize(None)
        new_value = float(data['energy'])
        print('energy: '+str(new_value)+'  '+'time: '+str(timestamp))
        if self.first_data_time == None:
            self.first_data_time = timestamp
            
        if timestamp-self.first_data_time > pd.Timedelta(2,'d'):
            if self.point_is_anomalous(new_value):
                print('An extreme point outlier just occured! \n\n\n\n')
                self.update(new_value)                   
                return True, {
                    "type": "extreme_value",
                    "sub_type": "",
                    "value": new_value,
                    "unit": "TODO"
                }
                    
        self.update(new_value)
        return False, {}
            
