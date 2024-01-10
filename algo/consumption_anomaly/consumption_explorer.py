import pandas as pd
from algo import utils
from . import consumption_utils

__all__ = ("Consumption_Explorer",)

class Consumption_Explorer:
    def __init__(self):

        self.first_data_time = None
        self.history_time_span = pd.Timedelta(14,'days')
        #TODO: Think about whether it's reasonable to make this a config.
        self.data_history = []

        self.step_until_next_estimate = pd.Timedelta(3, 'hours')
        #TODO: Think about whether it's reasonable to make this a config.

        self.last_estimate_time = None

        self.linear_model = None

        #TODO: Implement data persistence.

    def run(self, data):
        timestamp = utils.todatetime(data['energy_time']).tz_localize(None)
        new_value = float(data['energy'])
        print('energy: '+str(new_value)+'  '+'time: '+str(timestamp))
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

        if timestamp - self.first_data_time >= pd.Timedelta(2,'days'):
            anomaly_occured = consumption_utils.check_outlier(self.data_history, self.linear_model)
            if anomaly_occured:
                print('An anomaly in the total consumption curve just occured! \n\n\n\n')             
                return True, {
                    "type": "total_consumption_anomaly",
                    "sub_type": "",
                    "value": new_value,
                    "unit": "TODO"
                }