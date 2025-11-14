import pandas as pd
from algo import utils
from . import consumption_utils
import pickle
import os
import operator_lib.util as util

__all__ = ("Consumption_Explorer",)

class Consumption_Explorer:
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        self.first_data_time = None
        self.history_time_span = pd.Timedelta(14,'days')
        #TODO: Think about whether it's reasonable to make this a config.
        self.data_history = []

        self.step_until_next_estimate = pd.Timedelta(3, 'hours')
        #TODO: Think about whether it's reasonable to make this a config.

        self.last_estimate_time = None

        self.linear_model = None

        self.filename_dict = {"first_data_time": f'{data_path}/first_data_time.pickle', "data_history": f'{data_path}/data_history.pickle', 
                              "last_estimate_time": f'{data_path}/last_estimate_time.pickle', "linear_model": f'{data_path}/linear_model.pickle'}
        
        self.load_data()
    
    def save_data(self):
        first_data_time_path = self.filename_dict["first_data_time"]
        data_history_path = self.filename_dict["data_history"]
        last_estimate_time_path = self.filename_dict["last_estimate_time"]
        linear_model_path = self.filename_dict["linear_model"]

        with open(first_data_time_path, 'wb') as f:
            pickle.dump(self.first_data_time, f)
        with open(data_history_path, 'wb') as f:
            pickle.dump(self.data_history, f)
        with open(last_estimate_time_path, 'wb') as f:
            pickle.dump(self.last_estimate_time, f)
        with open(linear_model_path, 'wb') as f:
            pickle.dump(self.linear_model, f)

    def load_data(self):
        first_data_time_path = self.filename_dict["first_data_time"]
        data_history_path = self.filename_dict["data_history"]
        last_estimate_time_path = self.filename_dict["last_estimate_time"]
        linear_model_path = self.filename_dict["linear_model"]

        if os.path.exists(first_data_time_path):
            with open(first_data_time_path, 'rb') as f:
                self.first_data_time = pickle.load(f)
        if os.path.exists(data_history_path):
            with open(data_history_path, 'rb') as f:
                self.data_history = pickle.load(f)
        if os.path.exists(last_estimate_time_path):
            with open(last_estimate_time_path, 'rb') as f:
                self.last_estimate_time = pickle.load(f)
        if os.path.exists(linear_model_path):
            with open(linear_model_path, 'rb') as f:
                self.linear_model = pickle.load(f)

    def run(self, data):
        timestamp = utils.todatetime(data['time']).tz_localize(None)
        new_value = float(data['value'])
        util.logger.debug('value: '+str(new_value)+'  '+'time: '+str(timestamp))
        self.data_history = consumption_utils.update_data_history(timestamp, new_value, self.data_history, self.history_time_span)
        if self.first_data_time == None:
            self.first_data_time = timestamp
            return False, {}
        
        if self.last_estimate_time == None:
            if timestamp - self.first_data_time >= pd.Timedelta(2,'days'):
                self.linear_model, self.last_estimate_time = consumption_utils.update_linear_model(self.data_history)
        else:
            if timestamp - self.last_estimate_time >= self.step_until_next_estimate:
                self.linear_model, self.last_estimate_time = consumption_utils.update_linear_model(self.data_history)

        self.save_data()

        if timestamp - self.first_data_time >= pd.Timedelta(2,'days'):
            anomaly_occured = consumption_utils.check_outlier(self.data_history, self.linear_model)
            if anomaly_occured:
                util.logger.info('An anomaly in the total consumption curve just occured! \n\n\n\n')             
                return True, {
                    "type": "total_consumption_anomaly",
                    "sub_type": "",
                    "value": new_value,
                    "unit": "TODO"
                }