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
from math import floor, ceil

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#from ssl.new_SSL_TS_RP import temporal_shuffling, relative_positioning
from eegssl.preprocessing.new_preprocess import preprocess

MINUTES_TO_SECONDS = 60

class EEG_SSL_Dataset(Dataset):
    def __init__(self, T_pos, T_neg, anchor_windows_per_recording=2000,
    samples_per_anchor_window=6,
    raw_data_folder=None, preprocessed_file=None,
    save_preprocessed_path=None, sampling_freq=100,
    window_length=30, predict_delay=60):
        """
        Takes in either a data folder or a preprocessed file
        """
        self.T_pos = floor(T_pos * MINUTES_TO_SECONDS / window_length) # convert units from minutes to # of time windows
        self.T_neg = ceil(T_neg * MINUTES_TO_SECONDS / window_length)
        self.anchor_windows_per_recording = int(anchor_windows_per_recording)
        self.window_length = window_length # in seconds
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
        self.samples_per_anchor_window = samples_per_anchor_window

    def __len__(self):
        return self.num_files * self.anchor_windows_per_recording * self.samples_per_anchor_window

    def __getitem__(self, idx):
        file_idx = idx // (self.samples_per_anchor_window * self.anchor_windows_per_recording)
        sample_idx = idx % self.samples_per_anchor_window
        f = self.preprocessed[file_idx]
        anchor_idx = np.random.randint(len(f))

        ### Sampling with the indexes
        RP_sample, RP_label = self.relative_positioning(f, anchor_idx, sample_idx)

        return RP_sample, RP_label
    
    def sample_pos_idx(self, anchor_idx, recording_len):
        """
        sample positive sample index uniformly in the union of 2 intervals
        (anchor_idx-self.T_pos, anchor_idx) u (anchor_idx, anchor_idx+self.T_pos)
        """
        # TODO: check off by 1 errors
        left_interval_start = max(anchor_idx-self.T_pos + 1, 0) # (endpoint included)
        right_interval_end = min(anchor_idx+self.T_pos, recording_len) # (endpoint not included)
        
        random_idx = None
        while random_idx is None or random_idx == anchor_idx:
            random_idx = np.random.randint(left_interval_start, right_interval_end)
        return random_idx

    def sample_neg_idx(self, anchor_idx, recording_len):
        """
        sample negative sample index uniformly in the union of 2 intervals
        (0, anchor_idx-self.T_neg) U (anchor_idx+self.T_neg, recording_len)
        """
        # TODO: check off by 1 errors
        random_idx = np.random.randint(recording_len-2*self.T_neg)
        if random_idx >= anchor_idx - self.T_neg:
            random_idx += 2*self.T_neg
        return random_idx

    def relative_positioning(self, recording, anchor_idx, sample_idx):
        """ Retrives a self-supervised (relative positioning) sample
        Args:
            recording - Numpy datset of time-series arrays
            self.T_pos - positive context to sample from (in minutes)
            self.T_neg - negative context to sample from (in minutes)
            samples_per_anchor_window - int representing number of epochs to sample
        Output:
            RP_sample - Relative Positioning sample of dimensions (2, s, c)
                2 - sample1 + sample2
                s - # of eeg channels in each sample
                c - Samples per channel = 30s * 100Hz
            RP_labels - Relative Positioning label of dimensions (1)
                for each y = {1: if |sample1-sample2| < self.T_pos and -1: if |sample1-sample2| > self.T_neg}
        """
        anchor_window = recording[anchor_idx]
        if sample_idx < self.samples_per_anchor_window / 2: # self.T_pos loop
            pos_idx = self.sample_pos_idx(anchor_idx, len(recording))
            pos_window = recording[pos_idx]
            RP_sample = np.array([anchor_window, pos_window])
            RP_label = np.array([1])
        else: # Loop for self.T_neg
            neg_idx = self.sample_neg_idx(anchor_idx, len(recording))
            neg_window = recording[neg_idx]
            RP_sample = np.array([anchor_window, neg_window])
            RP_label = np.array([-1])
        return RP_sample, RP_label