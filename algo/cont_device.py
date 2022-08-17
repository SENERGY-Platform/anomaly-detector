import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import scipy.integrate as integrate
import similaritymeasures
from tqdm import tqdm

from . import error_calculation, preprocessing

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

def prepare_batches(history_data_series, batch_length_days, train_window_length=205):
    data_set_tr = preprocessing.minute_resampling(history_data_series)
    if history_data_series.index.max()-history_data_series.index.min() > pd.Timedelta(batch_length_days,'d'):
        data_set_tr = data_set_tr.loc[history_data_series.index.max()-pd.Timedelta(batch_length_days,'days'):]
    data_set_tr = preprocessing.smooth_data(data_set_tr)
    shift_dict_tr = {}
    for n in range(int(train_window_length/10)):
        shift_dict_tr[n] = data_set_tr[10*n:]
    return np.concatenate(tuple(preprocessing.decompose_into_time_windows(shift_dict_tr[n], train_window_length) for n in range(int(train_window_length/10))))

def batch_train(anomaly_detector, model_file_path, use_cuda, batch_length_days=50, epochs=10):
    autoencoder = anomaly_detector.model
    data_list = anomaly_detector.data
    data_series = pd.Series(data=[data_point for _, data_point in data_list], index=[timestamp.replace(microsecond=0) for timestamp, _ in data_list]).sort_index()
    data_series = data_series[~data_series.index.duplicated(keep='first')]
    normalized_history_data_series = preprocessing.normalize_data(data_series)
    training_batches = prepare_batches(normalized_history_data_series, batch_length_days, train_window_length=205)
    autoencoder, average_tr_loss_per_epoch_list = train(autoencoder, torch.Tensor(training_batches), epochs, use_cuda)
    anomaly_detector.training_performance.append(average_tr_loss_per_epoch_list)
    torch.save(autoencoder.state_dict(), model_file_path)
    return autoencoder

'''def get_area_errors(data_array, model, use_cuda):
    errors = []
    model.eval()
    for data_series in data_array:
        model_input = torch.Tensor(data_series)
        if use_cuda:
            model_input = model_input.cuda()
        model_output = model(model_input)
        errors.append(integrate.simpson(abs(np.squeeze(model_output.detach().cpu().numpy())-data_series)).item(0))
    model.train()
    return errors'''

def get_curve_length_measure_errors(model_input_data_array, model, use_cuda):
    errors = []
    model.eval()
    for data_series in model_input_data_array:
        model_input = torch.Tensor(data_series)
        if use_cuda:
            model_input = model_input.cuda()
        model_output = model(model_input)
        x = np.linspace(0,int(len(data_series)/2)-1,int(len(data_series)/2))
        y_1 = np.squeeze(model_output.detach().cpu().numpy())[0:int(len(data_series)/2)]
        y_2 = data_series[0:int(len(data_series)/2)]
        errors.append(similaritymeasures.curve_length_measure(np.column_stack((x,y_1)), np.column_stack((x,y_2)), metric='canberra'))
        z_1 = np.squeeze(model_output.detach().cpu().numpy())[len(data_series)-int(len(data_series)/2):]
        z_2 = data_series[len(data_series)-int(len(data_series)/2):]
        errors.append(similaritymeasures.curve_length_measure(np.column_stack((x,z_1)), np.column_stack((x,z_2)), metric='canberra'))
    model.train()
    return errors

def get_dtw_errors(model_input_data_array, model, use_cuda):
    errors = []
    model.eval()
    for data_series in model_input_data_array:
        model_input = torch.Tensor(data_series)
        if use_cuda:
            model_input = model_input.cuda()
        model_output = model(model_input)
        x = np.linspace(0,int(len(data_series)/2)-1,int(len(data_series)/2))
        y_1 = np.squeeze(model_output.detach().cpu().numpy())[0:int(len(data_series)/2)]
        y_2 = data_series[0:int(len(data_series)/2)]
        errors.append(similaritymeasures.dtw(np.column_stack((x,y_1)), np.column_stack((x,y_2)), metric='canberra')[0])
        z_1 = np.squeeze(model_output.detach().cpu().numpy())[len(data_series)-int(len(data_series)/2):]
        z_2 = data_series[len(data_series)-int(len(data_series)/2):]
        errors.append(similaritymeasures.dtw(np.column_stack((x,z_1)), np.column_stack((x,z_2)), metric='canberra')[0])
    model.train()
    return errors

'''def get_anomalous_part(anomalous_time_window, model, use_cuda, short_time_length=50):
    array_of_short_parts = []
    array_of_approx_short_parts = []

    for start in range(0,len(anomalous_time_window),short_time_length):
        if start+short_time_length <= 405:
            array_of_short_parts.append(anomalous_time_window[start:start+short_time_length])
            if use_cuda:
                approx_time_window = model(torch.Tensor(anomalous_time_window).cuda()).detach().cpu().numpy()
            else:
                approx_time_window = model(torch.Tensor(anomalous_time_window)).detach().cpu().numpy()
            array_of_approx_short_parts.append(approx_time_window[0,start:start+short_time_length])
    array_of_short_parts = np.array(array_of_short_parts)
    array_of_approx_short_parts = np.array(array_of_approx_short_parts)

    errors = []       
    for i in range(len(array_of_short_parts)):
        errors.append(integrate.simpson(abs(array_of_short_parts[i]-array_of_approx_short_parts[i])).item(0))
    return array_of_short_parts[np.argmax(errors)]'''

def test(data_list, anomaly_detector, use_cuda, model_input_window_length=205):
    anomaly_detector.model.eval()
    data_series = pd.Series(data=[data_point for _, data_point in data_list], index=[timestamp.replace(microsecond=0) for timestamp, _ in data_list]).sort_index()
    data_series = data_series[~data_series.index.duplicated(keep='first')]
    data_series = preprocessing.normalize_data(data_series)
    data_series = preprocessing.minute_resampling(data_series)
    data_series_smooth = preprocessing.smooth_data(data_series)
    model_input_data_array = preprocessing.decompose_into_time_windows(data_series_smooth, model_input_window_length)
    #reconstruction_area_errors = get_area_errors(data_array, anomaly_detector.model, use_cuda)
    reconstruction_curve_length_measure_errors = get_curve_length_measure_errors(model_input_data_array, anomaly_detector.model, use_cuda)
    reconstruction_dtw_errors = get_dtw_errors(model_input_data_array, anomaly_detector.model, use_cuda)
    anomalous_reconstruction_indices = error_calculation.get_anomalous_indices(reconstruction_dtw_errors)+error_calculation.get_anomalous_indices(reconstruction_curve_length_measure_errors)
    if 2*model_input_data_array.shape[0]-1 in anomalous_reconstruction_indices:
        anomalous_time_window = data_series[-model_input_window_length:]
        anomalous_time_window_smooth = data_series_smooth[-model_input_window_length:]
        anomalous_time_window_smooth_short = data_series_smooth[-int(model_input_window_length/2):]
        anomaly_detector.anomalies.append((anomalous_time_window,
                                           anomalous_time_window_smooth,
                                           anomalous_time_window_smooth_short))
        print('An anomaly has just occurred!')
        anomaly_detector.model.train()
        return 1
    anomaly_detector.model.train()

    