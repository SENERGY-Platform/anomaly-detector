import pickle
import os
import torch

def save_data(filename_dict, last_training_time, training_performance, model, training_max, reconstruction_errors):
        last_training_time_path = filename_dict["last_training_time"]
        training_performance_path = filename_dict["training_performance"]
        model_path = filename_dict["model"]
        training_max_path = filename_dict["training_max"]
        reconstruction_errors_path = filename_dict["reconstruction_errors"]

        with open(last_training_time_path, 'wb') as f:
            pickle.dump(last_training_time, f)
        with open(training_performance_path, 'wb') as f:
            pickle.dump(training_performance, f)
        torch.save(model, model_path)
        with open(training_max_path, 'wb') as f:
            pickle.dump(training_max, f)
        with open(reconstruction_errors_path, 'wb') as f:
            pickle.dump(reconstruction_errors, f)


def load_data(filename_dict, last_training_time, training_performance, model, training_max, reconstruction_errors):
    last_training_time_path = filename_dict["last_training_time"]
    training_performance_path = filename_dict["training_performance"]
    model_path = filename_dict["model"]
    training_max_path = filename_dict["training_max"]
    reconstruction_errors_path = filename_dict["reconstruction_errors"]

    if os.path.exists(last_training_time_path):
       with open(last_training_time_path, 'rb') as f:
           last_training_time = pickle.load(f)

    if os.path.exists(training_performance_path):
       with open(training_performance_path, 'rb') as f:
           training_performance = pickle.load(f)

    if os.path.exists(model_path):
        model = torch.load(model_path)

    if os.path.exists(training_max_path):
       with open(training_max_path, 'rb') as f:
           training_max = pickle.load(f)

    if os.path.exists(reconstruction_errors_path):
       with open(reconstruction_errors_path, 'rb') as f:
           reconstruction_errors = pickle.load(f)

    return last_training_time, training_performance, model, training_max, reconstruction_errors