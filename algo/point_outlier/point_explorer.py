import numpy as np
import pandas as pd
from algo import utils

__all__ = ("Point_Explorer",)

class Point_Explorer():
    def __init__(self):
        self.current_stddev = 0
        self.current_mean = 0
        self.num_datepoints = 0

        self.first_data_time = None

    def run(self, data):
        timestamp = utils.todatetime(data['energy_time']).tz_localize(None)
        new_value = float(data['energy'])
        print('energy: '+str(new_value)+'  '+'time: '+str(timestamp))
        if self.first_data_time == None:
            self.first_data_time = timestamp
            self.current_stddev = utils.calculate_std(new_value, self.current_stddev, self.current_mean, self.num_datepoints)
            self.current_mean = utils.calculate_mean(new_value, self.current_mean, self.num_datepoints)
            self.num_datepoints += 1
            return False, ''
        else:
            if timestamp-self.first_data_time > pd.Timedelta(2,'d'):
                if np.absolute(new_value-self.current_mean) > 3*self.current_stddev:
                    print('An extreme point outlier just occured! \n\n\n\n')
                    self.current_stddev = utils.calculate_std(new_value, self.current_stddev, self.current_mean, self.num_datepoints)
                    self.current_mean = utils.calculate_mean(new_value, self.current_mean, self.num_datepoints)
                    self.num_datepoints += 1
                    return True, 'point_outlier_anomaly'
                else:
                    self.current_stddev = utils.calculate_std(new_value, self.current_stddev, self.current_mean, self.num_datepoints)
                    self.current_mean = utils.calculate_mean(new_value, self.current_mean, self.num_datepoints)
                    self.num_datepoints += 1
                    return False, ''
            else:
                self.current_stddev = utils.calculate_std(new_value, self.current_stddev, self.current_mean, self.num_datepoints)
                self.current_mean = utils.calculate_mean(new_value, self.current_mean, self.num_datepoints)
                self.num_datepoints += 1
                return False, ''
