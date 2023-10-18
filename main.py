from hw1 import Composer, ComposerDataset, MidiCriticDataProcessor, Critic
from midi2seq import process_midi_seq
import torch, glob
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
# ***************************************** Test code for critic *****************************************
is_critic = False
train2Save = False
epoch_train2save = 100

if is_critic:
    print(" **** Testing critic ...")
    print("......")
    processor = MidiCriticDataProcessor(data_directory='.')
    print("Loading critic data ...")
    train_loader, test_loader = processor.prepare_data()
    critic = Critic(load_trained=False)

    if train2Save: # Task 1: For training the predictor 
        print("Training composer to save ...")
        critic.train2save(train_loader, epochs=epoch_train2save)
    else: # Task 2: For testing the possible test code from professor
        print("Training critic ...for training purpose only")
        epoch_loss = []
        epoch = 1
        for i in range(epoch):
            for x in train_loader:
                batch_loss = critic.train(x)
            epoch_loss.append(batch_loss)
        
        print(f'Epoch {i}, Epoch Loss: {np.mean(epoch_loss)}')


else: # ***************************************** Test code for Composer *****************************************

    print(" **** Testing Composer ...")
    print("......")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    dir_ = '.'
    maxlen = 200

    all_midis = glob.glob(f'{dir_}/maestro-v1.0.0/2004/*.midi')
    good_music = process_midi_seq(all_midis=all_midis, datadir=dir_, n=10000, maxlen=maxlen)
    #
    if train2Save:
        print("Training composer to save ...")
        dataset = ComposerDataset(good_music, sequence_length=200)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        #
        cps = Composer(load_trained=False)
        cps.train2save(data_loader, epochs=epoch_train2save)
    else:
        print("Training composer ...for training purpose only")
        piano_seq = torch.from_numpy(good_music)
        loader = DataLoader(TensorDataset(piano_seq), shuffle=True, batch_size=32*4)

        epoch = 1
        epoch_loss = []
        cps = Composer()
        for i in range(epoch):
            for x in loader:
                batch_loss = cps.train(x[0].long())
                epoch_loss.append(batch_loss)
            print(f'Epoch {i}, Epoch Loss: {np.mean(epoch_loss)}')


