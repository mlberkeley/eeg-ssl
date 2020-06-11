import numpy as np
import random

import os
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pickle

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
    def __init__(self, T_pos, T_neg, anchor_windows_per_recording=2000, raw_data_folder=None, preprocessed_file=None,
    save_preprocessed_path=None, sampling_freq=100,
    window_length=30, predict_delay=60):
        """
        Takes in either a data folder or a preprocessed file
        """
        self.T_pos = int(T_pos)
        self.T_neg = int(T_neg)
        self.anchor_windows_per_recording = int(anchor_windows_per_recording)
        self.window_length = window_length
        self.predict_delay = predict_delay
        self.sampling_freq = sampling_freq
        if ((raw_data_folder is None and preprocessed_file is None)
            or (raw_data_folder is not None and preprocessed_file is not None)):
            raise ValueError("Dataset requires a preprocessed_file or a raw_data_folder")
        
        if raw_data_folder is not None:
            self.files = [f for f in os.listdir(raw_data_folder) if f.endswith("PSG.edf")]
            self.preprocessed = []
            for f in tqdm(self.files):
                full_path = os.path.join(raw_data_folder, f)
                pp_file = preprocess(full_path)
                self.preprocessed.append(pp_file)
            if save_preprocessed_path is not None:
                pickle.dump((self.preprocessed, self.files), open(save_preprocessed_path, 'wb'))

        elif preprocessed_file is not None:
            self.preprocessed, self.files = pickle.load(open(preprocessed_file, 'rb'))

        self.num_files = len(self.files)
        self.num_samples = 6

    def __len__(self):
        return self.num_files * self.anchor_windows_per_recording * self.num_samples

    def __getitem__(self, idx):
        file_idx = idx // (self.num_samples * self.anchor_windows_per_recording)
        sample_idx = idx % self.num_samples
        f = self.preprocessed[file_idx]
        epoch_idx = np.random.randint(len(f))

        ### Sampling with the indexes
        RP_sample, RP_label = self.relative_positioning(f, epoch_idx, sample_idx)

        return RP_sample, RP_label
    
    def relative_positioning(self, recording, epoch_idx, sample_idx):
        """ Retrives a self-supervised (relative positioning) sample
        Args:
            recording - Numpy datset of time-series arrays
            self.T_pos - positive context to sample from
            self.T_neg - negative context to sample from
            num_samples - int representing number of epochs to sample
        Output:
            RP_sample - Relative Positioning sample of dimensions (2, s, c)
                2 - sample1 + sample2
                s - # of eeg channels in each sample
                c - Samples per channel = 30s * 100Hz
            RP_labels - Relative Positioning label of dimensions (1)
                for each y = {1: if |sample1-sample2| < self.T_pos and -1: if |sample1-sample2| > self.T_neg}
        """

        # TODO: Ask Alfredo to explain what is going on here.
        sample1 = recording[epoch_idx]
        if sample_idx <= 2: # self.T_pos loop
            # np.random.seed(sample_idx)
            sample2_index = np.random.randint(max(epoch_idx-self.T_pos, 0), min(epoch_idx+self.T_pos, recording.shape[0]-1))
            while sample2_index == epoch_idx: # should not be the same (TODO: could fix the previous line so this doesn't happen)
                sample2_index = np.random.randint(max(epoch_idx-self.T_pos, 0), min(epoch_idx+self.T_pos, recording.shape[0]-1))
            sample2 = recording[sample2_index]
            y = np.array([1])
            RP_sample = np.array([sample1, sample2])
            RP_label = y
        else: # Loop for self.T_neg
            # np.random.seed(sample_idx)
            if epoch_idx-self.T_neg <= 0: # self.T_neg if (corners)
                sample2_index = np.random.randint(epoch_idx+self.T_neg, recording.shape[0])
            elif epoch_idx+self.T_neg >= recording.shape[0]: # take care of low == high
                sample2_index = np.random.randint(0, epoch_idx-self.T_neg)
            else:
                sample2_index_1 = np.random.randint(epoch_idx+self.T_neg, recording.shape[0])
                sample2_index_2 = np.random.randint(0, epoch_idx-self.T_neg)
                sample2_index = list([sample2_index_1, sample2_index_2])[int(random.uniform(0,1))] # why int(random.uniform(0,1))? That's always 0...
            sample2 = recording[sample2_index]
            y = np.array([-1])
            RP_sample = np.array([sample1, sample2])
            RP_label = y
        return RP_sample, RP_label