import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import scipy.integrate as integrate

import preprocessing
import error_calculation

use_cuda = torch.cuda.is_available()


class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 10, stride=5) # Size of each channel: 405/5-1=80
        self.conv2 = nn.Conv1d(16, 32, 10, stride=5)# Size of each channel: 80/5-1=15
        
        self.fc1 = nn.Linear(480, latent_dims)
        
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        x = x.view(1,1,405)
        x = F.relu(self.dropout(self.conv1(x)))
        x = F.relu(self.dropout(self.conv2(x)))
        
        x = x.view(-1,480)
        
        x = self.fc1(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.fc1 = nn.Linear(latent_dims, 480)
        self.convt1 = nn.ConvTranspose1d(32, 16, kernel_size=10, stride=5)
        self.convt2 = nn.ConvTranspose1d(16, 1, kernel_size=10, stride=5)
        
        self.dropout = nn.Dropout(p=0.5)
        

    def forward(self, z):
        z = F.relu(self.dropout(self.fc1(z)))
        
        z = z.view(-1,32,15)
        z = F.relu(self.convt1(z))
        z = self.convt2(z)
        z = z.view(-1,405)
        
        return z

class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(autoencoder, tr_data, val_data, epochs, use_cuda):
    if use_cuda:
        autoencoder = autoencoder.cuda()
    opt = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)
    average_tr_loss_per_epoch_list = []
    #average_val_loss_per_epoch_list = []
    for _ in range(epochs):
        list_of_tr_losses = []
        list_of_val_losses = []
        for x in tr_data:
            if use_cuda:
                x = x.cuda() # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            tr_loss = ((x - x_hat)**2).sum()
            tr_loss.backward()
            opt.step()
            list_of_tr_losses.append(tr_loss)
            
        average_tr_loss_per_epoch = np.mean([loss.detach().cpu().numpy() for loss in list_of_tr_losses])
        average_tr_loss_per_epoch_list.append(average_tr_loss_per_epoch)
            
        #autoencoder.eval()
        #for x in val_data:
            #if use_cuda:
                #x = x.cuda()
            #with torch.no_grad():
                #x_hat = autoencoder(x)
                #val_loss = ((x-x_hat)**2).sum()
                #list_of_val_losses.append(val_loss)
        
        #average_val_loss_per_epoch = np.mean([loss.cpu().numpy() for loss in list_of_val_losses])
        #average_val_loss_per_epoch_list.append(average_val_loss_per_epoch)
        
        #autoencoder.train()
    return autoencoder, average_tr_loss_per_epoch_list#, average_val_loss_per_epoch_list    
            

def prepare_batches(history_data_series, batch_length_days):
    if history_data_series.Index.max-history_data_series.Index.min > pd.Timedelta(batch_length_days,'d'):
        data_set_tr = preprocessing.minute_resampling(history_data_series)
        data_set_tr = data_set_tr.iloc[-60*24*batch_length_days:]
        data_set_tr = preprocessing.smooth_data(data_set_tr)
        return preprocessing.decompose_into_time_windows(data_set_tr, window_length=405)
    else:
        return preprocessing.decompose_into_time_windows(history_data_series, window_length=405)

def batch_train(autoencoder, history_data_series, model_file_path, batch_length_days=14, epochs=1000):
    normalized_history_data_series = preprocessing.normalize_data(history_data_series)
    training_batches = prepare_batches(normalized_history_data_series, batch_length_days)
    autoencoder, _ = train(autoencoder, torch.Tensor(training_batches), torch.Tensor(training_batches), epochs, use_cuda)
    torch.save(autoencoder.state_dict(), model_file_path)
    return autoencoder

def get_errors(data_array, model): # How do we handle cuda in the code?
    errors = []
    model.eval()
    for data_series in data_array:
        x = model(torch.Tensor(data_series).cuda())
        errors.append(integrate.simpson(abs(x.detach().cpu().numpy()-data_series)).item(0))
    model.train()
    return errors

def get_anomalous_part(anomalous_time_window, model, short_time_length=50):
    array_of_short_parts = []
    array_of_approx_short_parts = []

    for start in range(0,len(anomalous_time_window),short_time_length):
        if i+short_time_length <= 405:
            array_of_short_parts.append(anomalous_time_window[start:start+short_time_length])
            approx_time_window = model(torch.Tensor(anomalous_time_window).cuda()).detach().cpu().numpy()
            array_of_approx_short_parts.append(approx_time_window [0,start:start+short_time_length])
    array_of_short_parts = np.array(array_of_short_parts)
    array_of_approx_short_parts = np.array(array_of_approx_short_parts)

    errors = []       
    for i in range(len(array_of_short_parts)):
        errors.append(integrate.simpson(abs(array_of_short_parts[i]-array_of_approx_short_parts[i])).item(0))
    return array_of_short_parts[np.argmax(errors)]

def test(data_series, model):
    model.eval()
    data_series = preprocessing.minute_resampling(data_series)
    data_series = preprocessing.smooth_data(data_series)
    data_array = preprocessing.decompose_into_time_windows(data_series, window_length=405)
    reconstruction_errors = get_errors(data_array, model)
    anomalous_reconstruction_indices = error_calculation.get_anomalous_indices(reconstruction_errors)
    if len(reconstruction_errors)-1 in anomalous_reconstruction_indices:
        # Maybe check for anomalous window part here.
        return 'An anomaly might just have occured!'
    model.train()

    