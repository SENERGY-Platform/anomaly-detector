from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def update_data_history(timestamp, new_value, data_history, history_time_span):
    data_history.append((timestamp, new_value))
    
    if len(data_history) == 0:
        return data_history

    first_ts_of_history = data_history[0][0]
    if timestamp - first_ts_of_history > history_time_span:
        del data_history[0]

    return data_history

def update_linear_model(data_history):
    sample_array_x, sample_array_target = compute_x_and_y_axis(data_history)
    model = LinearRegression().fit(sample_array_x, sample_array_target)
    estimation_time = pd.Timestamp(data_history[-1][0])
    return model, estimation_time

def check_outlier(data_history, model):
    sample_array_x, sample_array_target = compute_x_and_y_axis(data_history)
    sample_array_target.flatten()
    predicted_target = model.predict(sample_array_x)
    error_list = [np.abs(predicted_target[i] - sample_array_target[i]) for i in range(len(sample_array_target))] 
    three_sigma = np.mean(error_list[:-1]) + 3*np.std(error_list[:-1])
    if error_list[-1] > three_sigma:
        return True
    return False

def compute_x_and_y_axis(data_history):
    first_ts_of_history = data_history[0][0]
    x_list = [(ts-first_ts_of_history).total_seconds() for ts, _ in data_history]
    sample_array_x = np.array(x_list).reshape((-1,1))
    sample_array_target = np.array([value] for _, value in data_history)
    return sample_array_x, sample_array_target