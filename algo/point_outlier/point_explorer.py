import numpy as np
import pandas as pd
from algo import utils

__all__ = ("Point_Explorer",)

class Point_Explorer():
    def __init__(self):
        self.current_mean = 0
        self.current_stddev = 0
        self.num_datepoints = 0

        self.first_data_time = None

    def compute_mean(self,new_value):
        return (self.num_datepoints*self.current_mean + new_value)/(self.num_datepoints + 1)
    
    def compute_std(self, new_value):
        return np.sqrt(self.num_datepoints/(self.num_datepoints + 1)*self.current_stddev**2 + self.num_datepoints/((self.num_datepoints + 1)**2)*(new_value - self.current_mean)**2)

    def run(self, data):
        timestamp = utils.todatetime(data['energy_time']).tz_localize(None)
        if self.first_data_time == None:
            self.first_data_time = timestamp
            return
        else:
            new_value = float(data['energy'])
            self.current_stddev = self.compute_std(new_value)
            self.current_mean = self.compute_mean(new_value)
            self.num_datepoints += 1
            if timestamp-self.first_data_time > pd.Timedelta(2,'d'):
                if np.absolute(new_value-self.current_mean) > 3*self.current_stddev:
                   print('An extreme point outlier just occured! \n\n\n\n')
                   return 'point_outlier_anomaly'
            else:
                return
