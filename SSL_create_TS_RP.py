import os
import numpy as np
import pandas as pd
import mne
import pickle
import random
import sys
from os import listdir
from os.path import isfile, join
from mne import preprocessing

from PPD2 import PPD2
from SSL_TS_RP import temporal_shuffling, relative_positioning

def create(file_folder, T_pos, T_neg):
    """ Runs the whole pipeline and saves Numpy datasets and labels.

    Args:
        file_folder: The folder of the mne data objects
        T_pos: (time) sampling hyperparameter
        T_neg: (time) sampling hyperparameter

    returns
        None: saves the datasets to a new file
    """
    onlyfiles = [f for f in listdir(file_folder) if isfile(join(file_folder, f))]

    SAMPLE_TIME = 30
    CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']

    for data_file in onlyfiles:

        data = PPD2(data_file, SAMPLE_TIME, CHANNELS)

        ### __mainTS__(data, T_pos, T_neg):
        TS_dataset, TS_labels = temporal_shuffling(data, T_pos, T_neg)
        np.save('SSL_TS/TS_dataset' + str(file_folder), TS_dataset)
        np.save('SSL_TS/TS_labels' + str(file_folder), TS_labels)

        ### __mainRP__(data, T_pos, T_neg):
        RP_dataset, RP_labels = relative_positioning(data, 3, 3)
        np.save('SSL_RP/RP_dataset_' + str(file_folder), RP_dataset)
        np.save('SSL_RP/RP_labels_' + str(file_folder), RP_labels)

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    create(*sys.argv[1:])

### AAV :)
