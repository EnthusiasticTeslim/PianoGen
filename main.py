# ***************************************** Test code for critic *****************************************
from hw1 import MidiCriticDataProcessor, Critic
print(" **** Testing critic ...")
processor = MidiCriticDataProcessor(data_directory='.')
print("Loading critic data ...")
train_loader, test_loader = processor.prepare_data()
critic = Critic(load_trained=False)
Train2Save = True
print("Training critic ...")
if Train2Save: # Task 1: For training the predictor 
    critic.train2save(train_loader, epochs=100)
else: # Task 2: For testing the possible test code from professor
    epoch = 50
    for i in range(epoch):
        for x in train_loader:
            critic.train(x)
#

# ***************************************** Test code for Composer *****************************************