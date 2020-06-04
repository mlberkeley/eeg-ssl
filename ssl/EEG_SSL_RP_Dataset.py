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
from eegssl.preprocessing.new_preprocess import preprocess


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
        RP_dataset, RP_labels = self.relative_positioning(f, epoch_idx, sample_idx)
        RP_dataset = RP_dataset[sample_idx]
        RP_labels = RP_labels[sample_idx]

        return RP_dataset, RP_labels


    def get_batch(self, batch_size):
        minibatch_RP = []
        maxRange = self.num_files * self.num_epochs * 6
        files = random.sample(range(maxRange), batch_size)
        for idx in files:
            RP_dataset, RP_labels = self.__getitem__(idx)
            minibatch_RP.append((RP_dataset, RP_labels))

        return minibatch_RP
    
    def relative_positioning(self, epochs, epoch_idx, sample_idx):
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
        RP_dataset = np.empty((1, 2, epochs.shape[1], 3867))
        RP_labels = np.empty((1, 1))
        counter = 0
        sample1 = epochs[epoch_idx]
        if sample_idx <= 2: # self.T_pos loop
            np.random.seed(sample_idx)
            sample2_index = np.random.randint(max(epoch_idx-self.T_pos, 0), min(epoch_idx+self.T_pos, epochs.shape[0]-1))
            while sample2_index == epoch_idx: # should not be the same
                sample2_index = np.random.randint(max(epoch_idx-self.T_pos, 0), min(epoch_idx+self.T_pos, epochs.shape[0]-1))
            sample2 = epochs[sample2_index]
            y = 1
            RP_sample = np.array([sample1, sample2])
            RP_dataset[counter] = RP_sample
            RP_labels[counter] = y
            counter += 1
        else: # Loop for self.T_neg
            np.random.seed(sample_idx)
            if epoch_idx-self.T_neg <= 0: # self.T_neg if (corners)
                sample2_index = np.random.randint(epoch_idx+self.T_neg, epochs.shape[0])
            elif epoch_idx+self.T_neg >= epochs.shape[0]: # take care of low == high
                sample2_index = np.random.randint(0, epoch_idx-self.T_neg)
            else:
                sample2_index_1 = np.random.randint(epoch_idx+self.T_neg, epochs.shape[0])
                sample2_index_2 = np.random.randint(0, epoch_idx-self.T_neg)
                sample2_index = list([sample2_index_1, sample2_index_2])[int(random.uniform(0,1))]
            sample2 = epochs[sample2_index]
            y = -1
            RP_sample = np.array([sample1, sample2])
            RP_dataset[counter] = RP_sample
            RP_labels[counter] = y
            counter += 1
        return RP_dataset, RP_labels