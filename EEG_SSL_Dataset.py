import numpy as np
import random
import torch
from torch import optim
from torch.utils import data

from train_helpers import normalize_one
from models import CPC_EEG

import os
import numpy as np
import sys
from os import listdir
from os.path import isfile, join

from ssl.SSL_TS_RP import temporal_shuffling, relative_positioning

class EEG_SSL_Dataset():
	def __init__(self, preprocessed, T_pos_RP, T_neg_RP, T_pos_TS, T_neg_TS,\
	 			sampling_freq=100, window_length=30, predict_delay=60, batch_size=128):

		self.T_pos_RP = int(T_pos_RP)
		self.T_neg_RP = int(T_neg_RP)
		self.T_pos_TS = int(T_pos_TS)
		self.T_neg_TS = int(T_neg_TS)
		self.n_predict_windows = n_predict_windows
		self.n_negatives = n_negatives
		self.batch_size = batch_size
		self.window_length = window_length
		self.predict_delay = predict_delay
		self.overlap = overlap
		self.sampling_freq = sampling_freq


	def sample_negative(self, recording, start_sample, n_samples):
		n_available_positions = recording.shape[1] - n_samples - 2*self.window_length
    random_indices = np.random.choice(n_available_positions, self.n_negatives)
    negative_samples = []
    for i in random_indices:
      if i < start_sample - self.window_length:
        negative_samples.append(recording[:, i:i+self.window_length])
      else:
        idx = i + window_length + n_samples
        negative_samples.append(recording[:, idx:idx+window_length])
    return negative_samples

	def get_RP_minibatch(train_data):
		"""
		Return list has [{
			context windows: [[],[],...],
			prediction windows: [[],[],...]
			negative windows: [[],[],...]
		}]
		"""
		epoch_data = data[0]
		file_name = data[1]

		### relative_positioning(data, T_pos, T_neg):
		RP_dataset, RP_labels = relative_positioning(epoch_data, T_pos_RP, T_neg_RP)

		# Return Dataset
		return RP_dataset, RP_labels


	def get_TS_minibatch():

		epoch_data = data[0]
		file_name = data[1]

		# If folder doesn't exist, then create it.
		if not os.path.isdir("SSL_TS"):
			os.makedirs("SSL_TS")
		if not os.path.isdir("SSL_RP/RP_dataset") or not os.path.isdir("SSL_TS/TS_labels"):
			os.makedirs("SSL_TS/TS_dataset")
			os.makedirs("SSL_TS/TS_labels")

		# temporal_shuffling(data, T_pos, T_neg):
		TS_dataset, TS_labels = temporal_shuffling(epoch_data, T_pos_TS, T_neg_TS)

		# Save Dataset
		np.save('SSL_TS/TS_dataset/' + file_name, TS_dataset)
		np.save('SSL_TS/TS_labels/' + file_name, TS_labels)



		# sample bs subjects with replacement
		context_time = (1 + (self.n_context_windows-1)*self.overlap)*self.window_length
		predict_time = (1 + (self.n_predict_windows-1)*self.overlap)*self.window_length

		sample_length = self.sampling_freq*(context_time + self.predict_delay + self.predict_time)

		subjects = random.choices(train_data, k=bs)
		minibatch = []
		for s in subjects:
			s_length = s.shape[1]
			start_position = np.random.randint(0, s_length-sample_length)
			context_window_start_times = np.arange(start_position,
																						start_position + context_time*self.sampling_freq - overlap*self.sampling_freq*self.window_length,
																						overlap*S_FREQ*window_length)
			predict_window_start_times = np.arange(start_position + S_FREQ*context_time + S_FREQ*predict_delay,
																						start_position + S_FREQ*context_time + S_FREQ*predict_delay + S_FREQ*predict_time - overlap*S_FREQ*window_length,
																						overlap*S_FREQ*window_length)
			context_windows = [s[:,int(c_time):int(c_time)+S_FREQ*window_length] for c_time in context_window_start_times]
			predict_windows = [s[:,int(p_time):int(p_time)+S_FREQ*window_length] for p_time in predict_window_start_times]
			negative_windows = [sample_negative(s, int(start_position), int(sample_length), int(n_negatives)) for i in range(len(predict_windows))]

			minibatch.append({
				"context_windows": [normalize_one(c) for c in context_windows],
				"predict_windows": [normalize_one(c) for c in predict_windows],
				"negative_windows": [normalize_one(c) for vec in negative_windows for c in vec]
			})

		return minibatch
