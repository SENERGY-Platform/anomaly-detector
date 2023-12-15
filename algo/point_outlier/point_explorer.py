import numpy as np
import pandas as pd
from algo import utils
import pickle
import os

__all__ = ("Point_Explorer",)

class Point_Explorer():
    def __init__(self, data_path):
        self.filename_dict = {"current_stddev": f'{data_path}/current_stddev_point.parquet', "current_mean": f'{data_path}/current_mean_point.pickle', 
                              "num_datepoints": f'{data_path}/num_datepoints_point.pickle', "first_data_time": f'{data_path}/first_data_time_point.pickle'}
        
        self.current_stddev = 0
        self.current_mean = 0
        self.num_datepoints = 0
        self.first_data_time = None

        (self.current_stddev, 
        self.current_mean, 
        self.num_datepoints, 
        self.first_data_time) = self.load_data(self.current_stddev, 
                                              self.current_mean, 
                                              self.num_datepoints, 
                                              self.first_data_time)

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
            
    def save(self):
        current_stddev_path = self.filename_dict["current_stddev"]
        current_mean_path = self.filename_dict["current_mean"]
        num_datepoints_path = self.filename_dict["num_datepoints"]
        first_data_time_path = self.filename_dict["first_data_time"]

        with open(current_stddev_path, 'wb') as f:
            pickle.dump(self.current_stddev, f)
        with open(current_mean_path, 'wb') as f:
            pickle.dump(self.current_mean, f)
        with open(num_datepoints_path, 'wb') as f:
            pickle.dump(self.num_datepoints, f)
        with open(first_data_time_path, 'wb') as f:
            pickle.dump(self.first_data_time, f)

    def load_data(self, current_stddev, current_mean, num_datepoints, first_data_time):
        current_stddev_path = self.filename_dict["current_stddev"]
        current_mean_path = self.filename_dict["current_mean"]
        num_datepoints_path = self.filename_dict["num_datepoints"]
        first_data_time_path = self.filename_dict["first_data_time"]
        
        if os.path.exists(current_stddev_path):
            with open(current_stddev_path, 'rb') as f:
                current_stddev = pickle.load(f)
        if os.path.exists(current_mean_path):
            with open(current_mean_path, 'rb') as f:
                current_mean = pickle.load(f)
        if os.path.exists(num_datepoints_path):
            with open(num_datepoints_path, 'rb') as f:
                num_datepoints = pickle.load(f)
        if os.path.exists(first_data_time_path):
            with open(first_data_time_path, 'rb') as f:
                first_data_time = pickle.load(f)
        
        return current_stddev, current_mean, num_datepoints, first_data_time
