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
from torch.utils.data import DataLoader, TensorDataset, Dataset
from model_base import ComposerBase, CriticBase

from google_drive_downloader import GoogleDriveDownloader

##*************************Task 0*********************#
"""
    1.  What is the average score of the generated music? 0.8567
    
    
    2.  One idea to improve the quality of your composed music is to combine the composer with the critic to form a GAN and train the GAN to obtain a better composer. What is the major difficulty to train such a GAN?
    
        Balancing the interaction between the generator and critic, coupled with the inherent challenges of ensuring musical coherence and navigating subjective evaluations.

        GANs are notorious for being hard to train due to the min-max game between the generator (composer) and discriminator (critic). The network might not converge or might end up in a state where the generator is not improving.

    3.  Discuss a possible solution that may overcome this major difficulty.

        Instead of solely relying on the critic's feedback, use reinforcement learning where the composer (generator) is rewarded based on the critic's feedback. This approach, known as the "Actor-Critic" method in RL, 
        can help in better exploration and exploitation of the music space. The composer can be thought of as the 'actor' trying to generate good music, while the critic acts as the 'critic', providing feedback.

        Also, integrating attention mechanisms into the GAN, especially for the composer, to capture long-term dependencies in the music sequences.
    """

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


## Helper functions
# Convert labels to one-hot encoding
def convert_labels(labels):
    converted = torch.zeros(labels.size(0), 2)
    converted[labels.view(-1) == 1, 0] = 1
    converted[labels.view(-1) == 0, 1] = 1
    return converted

# Convert one-hot encoding to labels
def convert_labels_to_original(converted):
    labels = torch.zeros(converted.size(0), dtype=torch.long)
    labels[converted[:, 0] == 1] = 1
    labels[converted[:, 1] == 1] = 0

    return labels

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
# Training data
class MidiCriticDataProcessor:

    def __init__(self, data_directory, maxlen=200, test_size=0.2, random_state=42, batch_size=32):
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
        # print(f"Shape of Training data: {features_train.shape}")
        # print(f"Shape of Training labels: {label_train.shape}")
        # print(f"Shape of Test data: {features_test.shape}")
        # print(f"Shape of Test labels: {label_test.shape}")

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
    def __init__(self, num_embeddings=382, embedding_dim=10, hidden_dim=128, num_layers=3, n_classes=2):
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
            print('load model from file ...')
            # full google drive link: https://drive.google.com/file/d/1PCdNigWgslTn6SuqKxsSY7iUhNF_gfb0/view?usp=share_link
            # https://drive.google.com/file/d/1PCdNigWgslTn6SuqKxsSY7iUhNF_gfb0/view?usp=share_link
            GoogleDriveDownloader.download_file_from_google_drive(file_id='1PCdNigWgslTn6SuqKxsSY7iUhNF_gfb0',
                                    dest_path='./critic.pth',
                                    unzip=True)
            state_dict = torch.load('critic.pth').state_dict()
            self.model.load_state_dict(state_dict)
            print('Model loaded')
            self.model.eval()
        else:
            self.epoch = 0
            self.epoch_train_loss = []
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

            self.model.train()

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
            outputs ^= 1 # flip the outputs, so that 1 is good music and 0 is bad music
                    
        return outputs  
    
    def validate(self, val_loader, model):
        """Evaluate the network on the entire validation set."""

        loss_accum = []
        model.eval()
        with torch.set_grad_enabled(False):

            for (feature, label) in val_loader:
                label = convert_labels(label) # convert labels to one-hot encoding
                feature, label = feature.to(self.device).long(), label.to(self.device)

                outputs = self.model(feature)

                loss = self.criterion(outputs, label)
                loss_accum.append(loss.item())


        return np.mean(loss_accum)  
    
    def train(self, x):
        '''
        Train the model on one batch of data
        :param x: train data. For critic training, x will be a tuple of two tensors (data, label). expect a batch of dataloader
        :return: (mean) loss of the model on the batch
        '''
        
        (feature, label) = x
        label = convert_labels(label) # convert labels to one-hot encoding
        feature, label = feature.to(self.device).long(), label.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(feature)
        loss = self.criterion(outputs, label) 
        loss.backward()
        self.optimizer.step()
                
        # print(f"Loss: {loss.item():.3f}")
        
        return loss.item()

    def train2save(self, x, epochs=10, batch_size=32, fold=5):
        '''
        Train the model on one batch of data
        :param x: train data. For critic training, x will be a tuple of two tensors (data, label). expect a batch of dataloader
        :return: (mean) loss of the model on the batch
        '''

        # split data for K-fold cross validation to avoid overfitting
        indices = list(range(len(x.dataset)))
        kf = KFold(n_splits=fold, shuffle=True)
        cv_index = 0
        index_list_train = []
        index_list_valid = []
        for train_indices, valid_indices in kf.split(indices):
            index_list_train.append(train_indices)
            index_list_valid.append(valid_indices)

            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(valid_indices)

            train_loader = DataLoader(x.dataset, batch_size=batch_size,
                                                       sampler=train_sampler,
                                                       shuffle=False)
            val_loader = DataLoader(x.dataset, batch_size=batch_size,
                                                     sampler=valid_sampler,
                                                     shuffle=False)

            print(f'Start training..........CV index: {cv_index}')
            epoch_train_loss = []
            for epoch in range(epochs):
                #if epoch > 0: # since we initialize when self.load_trained is False, we do not want to re-initialize
                self.model.train()
                for (feature, label) in train_loader:
                    label = convert_labels(label) # convert labels to one-hot encoding
                    feature, label = feature.to(self.device).long(), label.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(feature)
                    loss = self.criterion(outputs, label)
                    loss.backward()
                    self.optimizer.step()

                    epoch_train_loss.append(loss.item())
                
                # loss
                train_loss_avg = np.mean(epoch_train_loss)
                val_loss_avg = self.validate(val_loader, self.model)

                print(f"Epoch {epoch}/{epochs}, CV {cv_index}/{fold} Train Loss: {train_loss_avg:3f}, Val Loss: {val_loss_avg:3f}")

        
            cv_index += 1   # increment cv index
        
        torch.save(self.model, 'critick.pth') 
        print("Finished training ...Model saved")
        
        return None


##*************************Task 2*********************#
#   (Class "Composer" should be a subclass of the class ComposerBase. You must use the exact class name.) 
#   You should implement a multi-layer (2 or 3 layers) LSTM model in this class. When the compose member 
#   function is called, it should return a sequence of events. Randomness is require in the implementation 
#   of the compose function such that each call to the function should generate a different sequence. 
#   The function "seq2piano" in "midi2seq.py" can be used to convert the sequence into a midi object, 
#   which can be written to a midi file and played on a computer. Train the model as a language model 
#   (autoregression) using the downloaded piano plays.

class ComposerDataset(Dataset):
    def __init__(self, music_sequences, sequence_length):
        self.music_sequences = music_sequences
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.music_sequences) - self.sequence_length
    
    def __getitem__(self, idx):
        sequence = self.music_sequences[idx, 0:self.sequence_length]
        target = self.music_sequences[idx, 1:self.sequence_length+1]
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class ComposerLSTM(nn.Module):
    def __init__(self, num_embeddings=382, embedding_dim=20, hidden_dim=512, num_layers=3, output_size=382):
        super(ComposerLSTM, self).__init__()

        self.num_embeddings = num_embeddings # number of unique words in the vocabulary
        self.embedding_dim = embedding_dim #
        self.hidden_dim = hidden_dim # Hidden dimension
        self.num_layers = num_layers # Number of LSTM layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim).to(self.device)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=0.2).to(self.device)
        self.fc = nn.Linear(self.hidden_dim, output_size).to(self.device)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)
        #x, _ = self.lstm(x)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        x, _ = self.lstm(x, (h0, c0))
        x = self.fc(x)
        return x
    

class Composer(ComposerBase):
    def __init__(self, load_trained=False):
        '''
        :param load_trained
            If load_trained is True, load a trained model from a file.
            Should include code to download the file from Google drive if necessary.
            else, construct the model
        '''    
        self.num_embeddings = 382
        self.embedding_dim = 20
        self.hidden_dim = 512
        self.num_layers = 3
        self.output_size = 382
        self.load_trained = load_trained
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        #print("Using device:", self.device)
        self.model = ComposerLSTM(
                                    num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim, 
                                    hidden_dim=self.hidden_dim, num_layers=self.num_layers, 
                                    output_size=self.output_size).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        if self.load_trained:
            print('load model from file ...')
            # full google drive link: https://drive.google.com/file/d/1a_8t-0tILi9nf9EjdA6TAIjMC9V-0KIu/view?usp=share_link
            # https://drive.google.com/file/d/1a_8t-0tILi9nf9EjdA6TAIjMC9V-0KIu/view?usp=share_link
            GoogleDriveDownloader.download_file_from_google_drive(file_id='1a_8t-0tILi9nf9EjdA6TAIjMC9V-0KIu',
                                    dest_path='./composer.pth',
                                    unzip=True)
            state_dict = torch.load('composer.pth').state_dict()
            self.model.load_state_dict(state_dict)
            print('Model loaded')
            self.model.eval()
        else:
            self.epoch = 0
            self.epoch_train_loss = []
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

            self.model.train()

    def train(self, x):

        sequence_length = x.size()[1]
        sequence, target = x[:, 0:sequence_length-1], x[:, 1:sequence_length+1]
        
        sequence = sequence.to(self.device) # Move tensors to the device
        target = target.to(self.device)
  
        self.optimizer.zero_grad()
        outputs = self.model(sequence)
        #loss = self.criterion(outputs.view(-1, outputs.shape[2]), target.view(-1))
        loss = self.criterion(outputs.reshape(-1, outputs.shape[2]), target.reshape(-1))
        
        loss.backward()
        self.optimizer.step() 
                
        #print(f'Loss: {loss.item():.4f}')
                
        return loss.item()
    
    def validate(self, val_loader, model):
        """Evaluate the network on the entire validation set."""

        model.eval()
        total_loss = []
        with torch.set_grad_enabled(False):
            for sequence, target in val_loader:
                sequence, target = sequence.to(self.device), target.to(self.device)

                outputs = model(sequence)
                loss = self.criterion(outputs.view(-1, outputs.shape[2]), target.view(-1))

                total_loss.append(loss.item())

        return np.mean(total_loss)
    
    
    def train2save(self, x, fold=5, batch=64, epochs=10):
        '''
        Train the model on one batch of data
        :param x: train data. For composer training, x will be a tuple of two tensors (data, label). expect a batch of dataloader
        :return: (mean) loss of the model on the batch
        '''

        # split data for K-fold cross validation to avoid overfitting
        indices = list(range(len(x.dataset)))
        kf = KFold(n_splits=fold, shuffle=True)
        cv_index = 0
        index_list_train = []
        index_list_valid = []
        for train_indices, valid_indices in kf.split(indices):
            index_list_train.append(train_indices)
            index_list_valid.append(valid_indices)

            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(valid_indices)

            train_loader = DataLoader(x.dataset, batch_size=batch,
                                                       sampler=train_sampler,
                                                       shuffle=False)
            val_loader = DataLoader(x.dataset, batch_size=batch,
                                                     sampler=valid_sampler,
                                                     shuffle=False)

            print(f'Start training ...CV index: {cv_index}')
            #self.model.train()
            train_total_loss = []
            for epoch in range(epochs):
                #if epoch > 0: # since we initialize when self.load_trained is False, we do not want to re-initialize
                self.model.train()
                for sequence, target in train_loader:
                    sequence, target = sequence.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(sequence)
                    loss = self.criterion(outputs.view(-1, outputs.shape[2]), target.view(-1))
                    loss.backward()
                    self.optimizer.step()

                    train_total_loss.append(loss.item())
                
                train_avg_loss = np.mean(train_total_loss)
                valid_avg_loss = self.validate(val_loader, self.model)
                

                print(f"Epoch {epoch+1}/{epochs}, CV {cv_index}/{fold},  Train Loss: {train_avg_loss:.3f}, Val Loss: {valid_avg_loss:.3f}")

            cv_index += 1   # increment cv index
        
        torch.save(self.model, 'composer.pth') 
        print("Finished training ...Model saved")
        
        return None

    def compose(self, n, temperature=1.0):

        '''
        Generate a music sequence
        :param n: length of the sequence to be generated
        :return: the generated sequence
        '''
        
        generated_sequence = [np.random.randint(self.num_embeddings)]
        with torch.no_grad():
            for _ in range(n):
                input_tensor = torch.tensor([generated_sequence[-1]], dtype=torch.long).to(self.device)
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
                
                output = self.model(input_tensor)
                output = output.squeeze().div(temperature).exp()
                next_note = torch.multinomial(output, 1).item()
                
                generated_sequence.append(next_note)

        return np.array(generated_sequence)
    
