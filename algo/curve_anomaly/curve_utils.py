import pandas as pd
from algo import utils
from . import cont_device, load_device

__all__ = ("batch_train", "test")


def batch_train(data_list, first_data_time, last_training_time, device_type, model, use_cuda, training_performance, training_max):
        current_timestamp = utils.todatetime(data_list[-1][0]).tz_localize(None)
        if current_timestamp-last_training_time.tz_localize(None) >= pd.Timedelta(14, 'days'): 
            if device_type == 'cont_device':
                if last_training_time.tz_localize(None) == first_data_time.tz_localize(None):
                    model = cont_device.Autoencoder(32)
                    if use_cuda:
                        model = model.cuda()
                model, training_performance, training_max = cont_device.batch_train(data_list, model, use_cuda, training_performance)
            elif device_type == 'load_device':
                return last_training_time, model, training_performance, training_max
            last_training_time = current_timestamp
            return last_training_time, model, training_performance, training_max
        elif current_timestamp-last_training_time.tz_localize(None) < pd.Timedelta(14, 'days'):
            return last_training_time, model, training_performance, training_max

def test(data_list, first_data_time, last_training_time, device_type, model, use_cuda, anomalies, loads, init_median, reconstruction_errors, training_max):
        if device_type == 'cont_device' and last_training_time.tz_localize(None) > first_data_time.tz_localize(None):
            output, anomalies, reconstruction_errors = cont_device.test(data_list, model, use_cuda, anomalies, training_max, reconstruction_errors)
            return output, loads, anomalies, reconstruction_errors
        elif device_type == 'load_device':
            output, loads, anomalies = load_device.train_test(data_list, loads, anomalies, init_median)
            return output,  loads, anomalies, reconstruction_errors
        else:
            return None, loads, anomalies, reconstruction_errors