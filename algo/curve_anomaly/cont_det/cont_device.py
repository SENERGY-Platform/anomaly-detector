import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import similaritymeasures
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import IsolationForest
from statistics import median

from . import preprocessing

__all__ = ("notification_decision",)

use_cuda = torch.cuda.is_available()


class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 7, stride=3) # Size of each channel: (205-7)/3+1=67
        self.conv2 = nn.Conv1d(16, 32, 7, stride=3)# Size of each channel: (67-7)/3+1=21
        
        self.fc1 = nn.Linear(672, latent_dims)
        
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        x = x.view(1,1,205)
        x = F.relu(self.dropout(self.conv1(x)))
        x = F.relu(self.dropout(self.conv2(x)))
        
        x = x.view(-1,672)
        
        x = self.fc1(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.fc1 = nn.Linear(latent_dims, 672)
        self.convt1 = nn.ConvTranspose1d(32, 16, kernel_size=7, stride=3)
        self.convt2 = nn.ConvTranspose1d(16, 1, kernel_size=7, stride=3)
        
        self.dropout = nn.Dropout(p=0.4)
        

    def forward(self, z):
        z = F.relu(self.dropout(self.fc1(z)))
        
        z = z.view(-1,32,21)
        z = F.relu(self.convt1(z))
        z = self.convt2(z)
        z = z.view(-1,205)
        
        return z

class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(autoencoder, tr_data, epochs, use_cuda):
    if use_cuda:
        autoencoder = autoencoder.cuda()
    opt = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)
    average_tr_loss_per_epoch_list = []
    for _ in tqdm(range(epochs)):
        list_of_tr_losses = []
        for x in tr_data:
            if use_cuda:
                x = x.cuda()
            opt.zero_grad()
            x_hat = autoencoder(x)
            tr_loss = ((x - x_hat)**2).sum()
            tr_loss.backward()
            opt.step()
            list_of_tr_losses.append(tr_loss)
            
        average_tr_loss_per_epoch = np.mean([loss.detach().cpu().numpy() for loss in list_of_tr_losses])
        average_tr_loss_per_epoch_list.append(average_tr_loss_per_epoch)
            
    return autoencoder, average_tr_loss_per_epoch_list   

def prepare_batches(history_data_series, train_window_length=205):
    data_set_tr = preprocessing.minute_resampling(history_data_series)
    data_set_tr = preprocessing.smooth_data(data_set_tr)
    shift_dict_tr = {}
    for n in range(int(train_window_length/10)):
        shift_dict_tr[n] = data_set_tr[10*n:]
    return np.concatenate(tuple(preprocessing.decompose_into_time_windows(shift_dict_tr[n], train_window_length) for n in range(int(train_window_length/10))))

def batch_train(data_list, model, use_cuda, training_performance, batch_length_days=50, epochs=20):
    autoencoder = model
    data_series = pd.Series(data=[data_point for _, data_point in data_list], index=[timestamp.replace(microsecond=0) for timestamp, _ in data_list]).sort_index()
    data_series = data_series[~data_series.index.duplicated(keep='first')]
    if data_series.index.max()-data_series.index.min() > pd.Timedelta(batch_length_days,'d'):
        data_series = data_series.loc[data_series.index.max()-pd.Timedelta(batch_length_days,'days'):]
    training_max = data_series.max()
    normalized_history_data_series = preprocessing.normalize_data(data_series, training_max)
    training_batches = prepare_batches(normalized_history_data_series, train_window_length=205)
    autoencoder, average_tr_loss_per_epoch_list = train(autoencoder, torch.Tensor(training_batches), epochs, use_cuda)
    training_performance.append(average_tr_loss_per_epoch_list)
    return autoencoder, training_performance, training_max

def get_reconstruction_errors(model_input_data_array, model, use_cuda):
    errors = []
    model.eval()
    for data_series in model_input_data_array:
        model_input = torch.Tensor(data_series)
        if use_cuda:
            model_input = model_input.cuda()
        try:
            model_output = model(model_input)
        except RuntimeError:
            return [None], np.empty(0)
        errors.append(abs(model_output.detach().cpu().numpy()-data_series).sum()/205)
    model.train()
    return errors, model_output.detach().cpu().numpy()

def update_reconstruction_errors_with_new_model(model, data_list, use_cuda, training_max, batch_length_days=50):
    model.eval()
    data_series = pd.Series(data=[data_point for _, data_point in data_list], index=[timestamp.replace(microsecond=0) for timestamp, _ in data_list]).sort_index()
    data_series = data_series[~data_series.index.duplicated(keep='first')]
    data_series = preprocessing.normalize_data(data_series, training_max)
    data_series = preprocessing.minute_resampling(data_series)
    data_series = data_series.loc[data_series.index.max()-pd.Timedelta(batch_length_days,'days'):]
    data_series_smooth = preprocessing.smooth_data(data_series)
    model_input_data_array = np.array(data_series_smooth).reshape(1,-1)
    reconstruction_errors = get_reconstruction_errors(model_input_data_array, model, use_cuda)[0]
    model.train()
    print("Reconstruction errors updated with new model!")
    return reconstruction_errors

def test(data_list, model, use_cuda, anomalies, training_max, reconstruction_errors, model_input_window_length=205):
    model.eval()
    data_series = pd.Series(data=[data_point for _, data_point in data_list], index=[timestamp.replace(microsecond=0) for timestamp, _ in data_list]).sort_index()
    data_series = data_series[~data_series.index.duplicated(keep='first')]
    data_series = preprocessing.normalize_data(data_series, training_max)
    data_series = preprocessing.minute_resampling(data_series)
    data_series = data_series[-model_input_window_length-30:]
    data_series_smooth = preprocessing.smooth_data(data_series)
    model_input_data_array = np.array(data_series_smooth[-model_input_window_length:]).reshape(1,-1)
    new_reconstruction_error = get_reconstruction_errors(model_input_data_array, model, use_cuda)[0][0]
    reconstructed_curve = get_reconstruction_errors(model_input_data_array, model, use_cuda)[1].flatten()
    if new_reconstruction_error == None: # This happens if not enough data is collected yet!
        model.train()
        return None, anomalies, reconstruction_errors
    if reconstruction_errors == None:
        reconstruction_errors = [new_reconstruction_error]
    else:
        reconstruction_errors.append(new_reconstruction_error)
        if len(reconstruction_errors) > 350: # I.e. erros from ~ 50 days.
            del reconstruction_errors[0]
    array_of_errors = np.array(reconstruction_errors).reshape(-1,1)
    array_of_errors = array_of_errors[~np.isnan(array_of_errors)].reshape(-1,1)
    anomalous_reconstruction_errors = get_anomalous_indices(array_of_errors,0.005)
    
    if len(array_of_errors)-1 in anomalous_reconstruction_errors and not np.isnan(new_reconstruction_error):
        anomalous_time_window = data_series[-model_input_window_length:]
        anomalous_time_window_smooth = data_series_smooth[-model_input_window_length:]
        anomalies.append((anomalous_time_window,
                                           anomalous_time_window_smooth, reconstructed_curve))
        print('An anomalous reconstruction error has just occurred!')
        model.train()
        return 'cont_device_anomaly', anomalies, reconstruction_errors
    else:
        model.train()
        return None, anomalies, reconstruction_errors
    

def notification_decision(timestamp_last_anomaly, timestamp):
    if timestamp <= pd.Timedelta(30,'T') + timestamp_last_anomaly:
        anomaly_during_last_30_min = True
    else:
        anomaly_during_last_30_min = False
    timestamp_last_anomaly = timestamp
    return timestamp_last_anomaly, anomaly_during_last_30_min
    
def get_anomalous_indices(array_of_errors,contam):
    anomalous_indices = []
    anomalous_error_model = IsolationForest(contamination=contam).fit(array_of_errors)
    predictions = anomalous_error_model.predict(array_of_errors)
    for i in range(len(array_of_errors)):
        if predictions[i]==-1 and array_of_errors[i]>median(array_of_errors):
            anomalous_indices.append(i)
    return anomalous_indices
