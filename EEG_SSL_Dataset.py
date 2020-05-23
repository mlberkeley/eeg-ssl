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

		return TS_dataset, TS_labels
