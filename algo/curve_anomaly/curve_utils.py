import pandas as pd
from algo import utils
from . import cont_device, load_device

__all__ = ("batch_train", "test")


def batch_train(data_list, first_data_time, last_training_time, device_type, model, use_cuda, training_performance):
        current_timestamp = utils.todatetime(data_list[-1][0]).tz_localize(None)
        if current_timestamp-last_training_time >= pd.Timedelta(14, 'days'): 
            if device_type == 'cont_device':
                if last_training_time == first_data_time:
                    model = cont_device.Autoencoder(32)
                    if use_cuda:
                        model = model.cuda()
                model, training_performance = cont_device.batch_train(data_list, model, use_cuda, training_performance)
            elif device_type == 'load_device':
                return last_training_time, model, training_performance
            last_training_time = current_timestamp
            return last_training_time, model, training_performance
        elif current_timestamp-last_training_time < pd.Timedelta(14, 'days'):
            return last_training_time, model, training_performance

def test(data_list, first_data_time, last_training_time, device_type, model, use_cuda, anomalies, loads, init_median):
        if device_type == 'cont_device' and last_training_time > first_data_time:
            output, anomalies = cont_device.test(data_list, model, use_cuda, anomalies)
            return output, loads, anomalies
        elif device_type == 'load_device':
            output, loads, anomalies = load_device.train_test(data_list, loads, anomalies, init_median)
            return output,  loads, anomalies
        else:
            return None, loads, anomalies