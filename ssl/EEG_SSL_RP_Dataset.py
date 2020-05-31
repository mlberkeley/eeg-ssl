import numpy as np
import random

import os
import numpy as np
import sys
from os import listdir
from os.path import isfile, join

import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#from ssl.new_SSL_TS_RP import temporal_shuffling, relative_positioning
from preprocessing.new_preprocess import preprocess


class EEG_SSL_Dataset(Dataset):

    def __init__(self, data_folder, T_pos, T_neg, sampling_freq=100, window_length=30, predict_delay=60, batch_size=128):
        self.data_folder = data_folder
        self.T_pos = int(T_pos)
        self.T_neg = int(T_neg)
        self.batch_size = batch_size
        self.window_length = window_length
        self.predict_delay = predict_delay
        self.sampling_freq = sampling_freq
        self.files = [f for f in os.listdir(data_folder) if f.endswith("PSG.edf")]
        self.preprocessed = []
        for f in self.files:
            full_path = os.path.join(data_folder, f)
            pp_file = preprocess(full_path)
            self.preprocessed.append(pp_file)

        self.num_files = len(self.files)
        self.num_epochs = len(self.preprocessed[0])
        self.num_samples = 6

    def __len__(self):
        return self.num_files * self.num_epochs * self.num_samples


    def __getitem__(self, idx):
        file_idx = (idx // 6) // self.num_epochs
        epoch_idx = (idx // 6) % self.num_epochs
        sample_idx = idx % 6

        ### Sampling with the indexes
        f = self.preprocessed[file_idx]
        RP_dataset, RP_labels = self.relative_positioning(f, epoch_idx)
        RP_dataset = RP_dataset[sample_idx]
        RP_labels = RP_labels[sample_idx]

        return RP_labels, RP_labels


    def get_batch(self):
        # minibatch_TS = []
        # files = random.sample(self.files, int(num_users))
        # for f in files:
        #     full_path = os.path.join(data_folder, f)

        #     TS_dataset, TS_labels = temporal_shuffling(preprocessed)
        #     minibatch_TS.append((TS_dataset, TS_labels))

        # return minibatch_TS
        return None
    
    def relative_positioning(self, epochs, idx):
        """ Builds a self-supervised (relative positioning) dataset of epochs

        Args:
            epochs - Numpy datset of time-series arrays
            self.T_pos - positive context to sample from
            self.T_neg - negative context to sample from
            num_samples - int representing number of epochs to sample

        Output:
            TS_dataset - Temporal Shuffling Dataset of dimensions (L, 3, s, c)
                L - # of samples = # of user * # of epochs per user * 6
                2 - sample1 + sample2
                s - # of eeg channels in each sample
                c - Samples per channel = 30s * 128Hz
            TS_labels - Temporal Shuffling labels of dimensions (1, L)
                for each y = {1: if |sample1-sample2| < self.T_pos and -1: if |sample1-sample2| > self.T_neg}
        """
        np.random.seed(0)
        total_samples = self.num_samples
        RP_dataset = np.empty((6, 2, epochs.shape[1], 3867))   #why 6?
        RP_labels = np.empty((6, 1))
        counter = 0

        #select random epoch and get index
        print("\n\n\n EPOCHS:")
        print(epochs)


        sample1 = epochs[idx]
        for _ in range(3): # Loop for self.T_pos
            sample2_index = np.random.randint(max(idx-self.T_pos, 0), min(idx+self.T_pos, epochs.shape[0]-1))
            while sample2_index == idx: # should not be the same
                sample2_index = np.random.randint(max(idx-self.T_pos, 0), min(idx+self.T_pos, epochs.shape[0]-1))
            sample2 = epochs[sample2_index]

            y = 1


            RP_sample = np.array([sample1, sample2])
            RP_dataset[counter] = RP_sample
            RP_labels[counter] = y
            counter += 1

        for _ in range(3): # Loop for self.T_neg
            if idx-self.T_neg <= 0: # self.T_neg if (corners)
                sample2_index = np.random.randint(idx+self.T_neg, epochs.shape[0])
            elif idx+self.T_neg >= epochs.shape[0]: # take care of low == high
                sample2_index = np.random.randint(0, idx-self.T_neg)
            else:
                sample2_index_1 = np.random.randint(idx+self.T_neg, epochs.shape[0])
                sample2_index_2 = np.random.randint(0, idx-self.T_neg)
                sample2_index = list([sample2_index_1, sample2_index_2])[int(random.uniform(0,1))]
            sample2 = epochs[sample2_index]

            y = -1

            RP_sample = np.array([sample1, sample2])
            RP_dataset[counter] = RP_sample
            RP_labels[counter] = y
            counter += 1

        return RP_dataset, RP_labels