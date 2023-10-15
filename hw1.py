# Author:   Teslim Olayiwola
# ID:       890284015

# Importing libraries
import numpy as np
import os, sys, time, datetime, pickle, copy, random, glob, logging

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
class MidiDataProcessor:

    def __init__(self, data_directory, maxlen=100, test_size=0.2, random_state=42, batch_size=32):
        self.data_directory = data_directory
        self.maxlen = maxlen
        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size

    def __get__(self, idx):
        return self.all_data[idx], self.all_labels[idx]

    def prepare_data(self, is_scale=False):
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
        features_train = features_train.reshape((-1, features_train.shape[1], 1))
        features_test = features_test.reshape((-1, features_test.shape[1], 1))

        label_train = label_train.reshape((-1, 1))
        label_test = label_test.reshape((-1, 1))

        def convert_labels(labels):
            converted = torch.zeros(labels.size(0), 2)
            converted[labels.view(-1) == 1, 0] = 1
            converted[labels.view(-1) == 0, 1] = 1
            return converted
        
        label_train = convert_labels(label_train)
        label_test = convert_labels(label_test)
        
        if is_scale:
            scaler = MinMaxScaler()
            features_train = scaler.fit_transform(features_train)
            features_test = scaler.transform(features_test)

        features_train = torch.Tensor(features_train).to(device)
        features_test = torch.Tensor(features_test).to(device)
        label_train = torch.Tensor(label_train).to(device)
        label_test = torch.Tensor(label_test).to(device)

        train_dataset = TensorDataset(features_train, label_train)
        test_dataset = TensorDataset(features_test, label_test)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=self.batch_size)

        return train_loader, test_loader

    def __repr__(self):
        return f'MidiDataProcessor(data_directory={self.data_directory!r}, maxlen={self.maxlen}, test_size={self.test_size}, random_state={self.random_state}, batch_size={self.batch_size})'

# Critic model
class LSTMCritic(nn.Module):
    def __init__(self, input_dim=1, hidden_size=64, num_layers=3, n_classes=2):
        super(LSTMCritic, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = n_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.lstm = nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        batch_size = x.size(0)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # linear layer
        out = self.fc(out)
        return out

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
        self.model = LSTMCritic(input_dim=1, hidden_size=64, num_layers=3, n_classes=2).to(self.device)
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

    def train(self, x, epochs=10, lr=1e-5):
        '''
        Train the model on one batch of data
        :param x: train data. For critic training, x will be a tuple of two tensors (data, label). expect a batch of dataloader
        :return: (mean) loss of the model on the batch
        '''
            
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        logging.info('Start training ...')
        self.model.train()
        total_loss = 0
        for epoch in range(epochs):
            for idx, (feature, label) in enumerate(x):
                feature, label = feature.to(self.device), label.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(feature)
                loss = self.criterion(outputs, label)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        
        logging.info("Finished training ...")

        torch.save(self.model, 'critic.pth')
        
        return total_loss/feature.size(0)

##*************************Task 2*********************#
#   (Class "Composer" should be a subclass of the class ComposerBase. You must use the exact class name.) 
#   You should implement a multi-layer (2 or 3 layers) LSTM model in this class. When the compose member 
#   function is called, it should return a sequence of events. Randomness is require in the implementation 
#   of the compose function such that each call to the function should generate a different sequence. 
#   The function "seq2piano" in "midi2seq.py" can be used to convert the sequence into a midi object, 
#   which can be written to a midi file and played on a computer. Train the model as a language model 
#   (autoregression) using the downloaded piano plays.
