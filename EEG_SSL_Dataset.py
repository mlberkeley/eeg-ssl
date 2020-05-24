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

from ssl.SSL_TS_RP import temporal_shuffling, relative_positioning
from preprocessing.new_preprocess import preprocess

class EEG_SSL_Dataset():
	def __init__(self, preprocessed, T_pos_RP, T_neg_RP, T_pos_TS, T_neg_TS,\
	 			sampling_freq=100, window_length=30, predict_delay=60, batch_size=128):

		self.T_pos_RP = int(T_pos_RP)
		self.T_neg_RP = int(T_neg_RP)
		self.T_pos_TS = int(T_pos_TS)
		self.T_neg_TS = int(T_neg_TS)
		#self.n_predict_windows = n_predict_windows
		#self.n_negatives = n_negatives
		self.batch_size = batch_size
		self.window_length = window_length
		self.predict_delay = predict_delay
		#self.overlap = overlap
		self.sampling_freq = sampling_freq

	def get_RP_minibatch(self, data_folder, T_pos_RP, T_neg_RP, num_users):
		"""
		
		"""
		minibatch_RP = []
		files = random.sample([f for f in os.listdir(data_folder) if f.endswith("PSG.edf")], int(num_users))
		for f in files:
			full_path = os.path.join(data_folder, f)
			preprocessed = preprocess(full_path)
			RP_dataset, RP_labels = relative_positioning(preprocessed, int(T_pos_RP), int(T_neg_RP))
			minibatch_RP.append((RP_dataset, RP_labels))
			
		return minibatch_RP


	def get_TS_minibatch(self, data_folder, T_pos_TS, T_neg_TS, num_users):
		"""
		
		"""
		minibatch_TS = []
		files = random.sample([f for f in os.listdir(data_folder) if f.endswith("PSG.edf")], int(num_users))
		for f in files:
			full_path = os.path.join(data_folder, f)
			preprocessed = preprocess(full_path)
			TS_dataset, TS_labels = temporal_shuffling(preprocessed, int(T_pos_TS), int(T_neg_TS))
			minibatch_TS.append((TS_dataset, TS_labels))
			
		return minibatch_TS
		
