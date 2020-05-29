import numpy as np
import random
#import torch
#from torch import optim
#from torch.utils import data

#from train_helpers import normalize_one
#from models import CPC_EEG

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

plt.ion()

from ssl.new_SSL_TS_RP import temporal_shuffling, relative_positioning
from preprocessing.new_preprocess import preprocess


class EEG_SSL_RP_Dataset(Dataset):


	def __init__(self, preprocessed, T_pos_RP, T_neg_RP,\
	 			sampling_freq=100, window_length=30, predict_delay=60, batch_size=128):

		self.T_pos_RP = int(T_pos_RP)
		self.T_neg_RP = int(T_neg_RP)

		self.batch_size = batch_size
		self.window_length = window_length
		self.predict_delay = predict_delay
		self.sampling_freq = sampling_freq


	def get_RP_minibatch(data_folder, T_pos_RP, T_neg_RP, num_users, num_samples):
	"""

	"""
	minibatch_RP = []
	files = random.sample([f for f in os.listdir(data_folder) if f.endswith("PSG.edf")], int(num_users))
	for f in files:
		full_path = os.path.join(data_folder, f)
		preprocessed = preprocess(full_path)
		RP_dataset, RP_labels = relative_positioning(preprocessed, int(T_pos_RP), int(T_neg_RP), int(num_samples))
		minibatch_RP.append((RP_dataset, RP_labels))

	return minibatch_RP


	def relative_positioning(epochs, T_pos, T_neg, num_samples):
	    """ Builds a self-supervised (relative positioning) dataset of epochs

	    Args:
	        epochs - Numpy datset of time-series arrays
	        T_pos - positive context to sample from
	        T_neg - negative context to sample from
	        num_samples - int representing number of epochs to sample

	    Output:
	        TS_dataset - Temporal Shuffling Dataset of dimensions (L, 3, s, c)
	            L - # of samples = # of user * # of epochs per user * 6
	            2 - sample1 + sample2
	            s - # of eeg channels in each sample
	            c - Samples per channel = 30s * 128Hz
	        TS_labels - Temporal Shuffling labels of dimensions (1, L)
	            for each y = {1: if |sample1-sample2| < T_pos and -1: if |sample1-sample2| > T_neg}
	    """
	    total_samples = epochs.shape[0] * 6
	    RP_dataset = np.empty((total_samples, 2, epochs.shape[1], 3867)) ##TODO change shape of dataset according to num_samples
	    RP_labels = np.empty((total_samples, 1)) ##TODO change shape of dataset according to num_samples
	    counter = 0

	    #select random epoch and get index
	    print("\n\n\n EPOCHS:")
	    print(epochs)

	    for _, idx in enumerate(np.random.randint(len(epochs), size = num_samples)):
	        sample1 = epochs[idx]
	        for _ in range(3): # Loop for T_pos
	            sample2_index = np.random.randint(max(idx-T_pos, 0), min(idx+T_pos, epochs.shape[0]-1))
	            while sample2_index == idx: # should not be the same
	                sample2_index = np.random.randint(max(idx-T_pos, 0), min(idx+T_pos, epochs.shape[0]-1))
	            sample2 = epochs[sample2_index]

	            y = 1

	            RP_sample = np.array([sample1, sample2])
	            RP_dataset[counter] = RP_sample
	            RP_labels[counter] = y
	            counter += 1

	        for _ in range(3): # Loop for T_neg
	            if idx-T_neg <= 0: # T_neg if (corners)
	                sample2_index = np.random.randint(idx+T_neg, epochs.shape[0])
	            elif idx+T_neg >= epochs.shape[0]: # take care of low == high
	                sample2_index = np.random.randint(0, idx-T_neg)
	            else:
	                sample2_index_1 = np.random.randint(idx+T_neg, epochs.shape[0])
	                sample2_index_2 = np.random.randint(0, idx-T_neg)
	                sample2_index = list([sample2_index_1, sample2_index_2])[int(random.uniform(0,1))]
	            sample2 = epochs[sample2_index]

	            y = -1

	            RP_sample = np.array([sample1, sample2])
	            RP_dataset[counter] = RP_sample
	            RP_labels[counter] = y
	            counter += 1

	    return RP_dataset, RP_label
