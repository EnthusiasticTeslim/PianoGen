# Author:   Teslim Olayiwola
# ID:       890284015

# Importing libraries
import numpy as np
import os, sys, time, datetime, pickle, copy, random, glob, logging

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from midi2seq import piano2seq, random_piano, process_midi_seq
from torch.utils.data import DataLoader, TensorDataset
from model_base import ComposerBase, CriticBase

from google_drive_downloader import GoogleDriveDownloader as gdd

##*************************Task 1*********************#
#
#   (Class "Critic" should be a subclass of the class CriticBase. You must use the exact class name.) 
#   You should implement a multi-layer (2 or 3 layers) LSTM model in this class. 
#   The Model (the score function) takes a sequence of envents as input and outputs a score judging 
#   whether the piano music corresponding to the sequence is good music or bad music. 
#   A function to generate random music is provided in the "midi2seq.py". 
#   Use the function to create a collection of random piano plays as examples of bad music. 
#   Use the piano plays in the downloaded data as example of good music. 
#   (You don't need to use all the downloaded data. A sufficiently large subset will be enough.) 
#   Train the model in this class using both the good and the bad examples.


## Training data
## Training data
def convert_labels(labels):
    converted = torch.zeros(labels.size(0), 2)
    converted[labels.view(-1) == 1, 0] = 1
    converted[labels.view(-1) == 0, 1] = 1
    return converted
class MidiDataProcessor:

    def __init__(self, data_directory, maxlen=100, test_size=0.2, random_state=42, batch_size=32):
        self.data_directory = data_directory
        self.maxlen = maxlen
        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size

    def __get__(self, idx):
        return self.all_data[idx], self.all_labels[idx]

    def prepare_data(self):
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        all_midis = glob.glob(f'{self.data_directory}/maestro-v1.0.0/**/*.midi')

        good_music_midi = process_midi_seq(all_midis=all_midis, datadir=self.data_directory, n=10000, maxlen=self.maxlen)
        bad_music_midi = [random_piano(n=self.maxlen) for _ in range(len(all_midis))]
        bad_music_midi = process_midi_seq(all_midis=bad_music_midi, datadir=self.data_directory, n=10000, maxlen=self.maxlen)

        good_music = torch.tensor(good_music_midi, dtype=torch.float32)
        bad_music = torch.tensor(bad_music_midi, dtype=torch.float32)

        good_labels = torch.ones((len(good_music), 1))
        bad_labels = torch.zeros((len(bad_music), 1))

        self.all_data = torch.cat([good_music, bad_music], dim=0)
        self.all_labels = torch.cat([good_labels, bad_labels], dim=0)

        features_train, features_test, label_train, label_test = train_test_split(
                                                                                    self.all_data, self.all_labels,
                                                                                    test_size=self.test_size,
                                                                                    random_state=self.random_state,
                                                                                    shuffle=True)
        
        # label_train = convert_labels(label_train)
        # label_test = convert_labels(label_test)

        features_train = torch.Tensor(features_train).to(device)
        features_test = torch.Tensor(features_test).to(device)

        label_train = torch.Tensor(label_train).to(device)
        label_test = torch.Tensor(label_test).to(device)

        train_dataset = TensorDataset(features_train, label_train)
        test_dataset = TensorDataset(features_test, label_test)

        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_dataset, shuffle=True, batch_size=self.batch_size)

        return self.train_loader, self.test_loader

    def __repr__(self):
        return f'MidiDataProcessor(data_directory={self.data_directory!r}, maxlen={self.maxlen}, test_size={self.test_size}, random_state={self.random_state}, batch_size={self.batch_size}, train_loader size={len(self.train_loader.dataset)}, test_loader size={len(self.test_loader.dataset)})'

# Critic model
# n x max_len -> embedding -> n x nax_len x max_len - 1 -> LSTM (with hidden size=3)-> n x 2
class LSTMCritic(nn.Module):
    def __init__(self, num_embeddings=382, embedding_dim=100, hidden_dim=128, num_layers=3, n_classes=2):
        super(LSTMCritic, self).__init__()
        self.num_embeddings = num_embeddings # number of unique words in the vocabulary
        self.embedding_dim = embedding_dim #
        self.hidden_dim = hidden_dim # Hidden dimension
        self.num_layers = num_layers # Number of LSTM layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim).to(self.device)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=0.2).to(self.device)
        self.fc = nn.Linear(self.hidden_dim, n_classes).to(self.device)
        

    def forward(self, x):
        # Embedding layer
        x = self.embedding(x).to(self.device)
        # LSTM forward pass
        batch_size = x.size(0)
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = lstm_out[:, -1, :]
        # Linear layer
        out = self.fc(out)
        return out

# Accumulation meter
class AccumulationMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.avg = 0.0
        self.sum = 0
        self.count = 0.0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        self.sqrt = self.value ** 0.5
        self.rmse = self.avg ** 0.5

# Early stoppping criteria
class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
# Critic class
class Critic(CriticBase):
    def __init__(self, load_trained=False):
        '''
        :param load_trained
            If load_trained is True, load a trained model from a file.
            Should include code to download the file from Google drive if necessary.
            else, construct the model
        '''    
    
        self.load_trained = load_trained
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = LSTMCritic().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        if self.load_trained:
            logging.info('load model from file ...')
            gdd.download_file_from_google_drive(file_id='18YkTrsqa0dWCVC4PpE2_7q8nN3jxdzhD',
                                    dest_path='./critic.pth',
                                    unzip=True)
            self.model = torch.load('critic.pth')
            self.model.eval()

    def score(self, x):
        '''
        Compute the score of a music sequence
        :param x: a music sequence
        :return: the score between 0 and 1 that reflects the quality of the music: the closer to 1, the better
        '''
        with torch.set_grad_enabled(False):
            logging.info('Compute score ...')
        
            outputs = self.model(x.to(self.device))
            outputs = torch.argmax(outputs, dim=1)
            outputs ^= 1 # index 0 is good and index 1 is bad 
                    
        return outputs  
    
    def validate(self, val_loader, model):
        """Evaluate the network on the entire validation set."""

        loss_accum = AccumulationMeter()
        model.eval()
        with torch.set_grad_enabled(False):

            for i, (feature, label) in enumerate(val_loader):
                label = convert_labels(label) # convert labels to one-hot encoding
                feature, label = feature.to(self.device).long(), label.to(self.device)

                outputs = self.model(feature)

                loss = self.criterion(outputs, label)
                loss_accum.update(loss.item(), label.size(0))
 

        return loss_accum.rmse

    def train(self, x, epochs=10, lr=1e-5):
        '''
        Train the model on one batch of data
        :param x: train data. For critic training, x will be a tuple of two tensors (data, label). expect a batch of dataloader
        :return: (mean) loss of the model on the batch
        '''
            
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_accum_train = AccumulationMeter()

        # split data for K-fold cross validation to avoid overfitting
        indices = list(range(len(x.dataset)))
        kf = KFold(n_splits=5, shuffle=True)
        cv_index = 0
        index_list_train = []
        index_list_valid = []
        for train_indices, valid_indices in kf.split(indices):
            index_list_train.append(train_indices)
            index_list_valid.append(valid_indices)

            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(valid_indices)

            train_loader = DataLoader(x.dataset, batch_size=32,
                                                       sampler=train_sampler,
                                                       shuffle=False)
            val_loader = DataLoader(x.dataset, batch_size=32,
                                                     sampler=valid_sampler,
                                                     shuffle=False)

            logging.info('Start training ...')
            self.model.train()
            early_stopping = EarlyStopping(tolerance=5, min_delta=10)
            epoch_train_loss = []
            epoch_validate_loss = []
            for epoch in range(epochs):
                for idx, (feature, label) in enumerate(train_loader):
                    label = convert_labels(label) # convert labels to one-hot encoding
                    feature, label = feature.to(self.device).long(), label.to(self.device)

                    outputs = self.model(feature)
                    loss = self.criterion(outputs, label)
                    #total_loss += loss.item()
                    loss_accum_train.update(loss.item(), label.size(0))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    torch.backends.cudnn.enabled = False
                # training loss
                epoch_train_loss.append(loss_accum_train.avg)
                
                # validation loss
                val_loss_avg = self.validate(val_loader, self.model)
                epoch_validate_loss.append(val_loss_avg)

                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss_accum_train.avg}, Val Loss: {val_loss_avg}")

                # early stopping
                early_stopping(loss_accum_train.avg, val_loss_avg)
                if early_stopping.early_stop:
                    print("We are at epoch:", epoch)
                    logging.info("Finished training ...Model saved")
                    torch.save(self.model, 'critic.pth') 
                    break
        
            cv_index += 1   # increment cv index
        
        return loss_accum_train.avg

# Test code

processor = MidiDataProcessor(data_directory='.')
train_loader, test_loader = processor.prepare_data()

critic = Critic(load_trained=False)
critic.train(train_loader, epochs=100)

##*************************Task 2*********************#
#   (Class "Composer" should be a subclass of the class ComposerBase. You must use the exact class name.) 
#   You should implement a multi-layer (2 or 3 layers) LSTM model in this class. When the compose member 
#   function is called, it should return a sequence of events. Randomness is require in the implementation 
#   of the compose function such that each call to the function should generate a different sequence. 
#   The function "seq2piano" in "midi2seq.py" can be used to convert the sequence into a midi object, 
#   which can be written to a midi file and played on a computer. Train the model as a language model 
#   (autoregression) using the downloaded piano plays.
