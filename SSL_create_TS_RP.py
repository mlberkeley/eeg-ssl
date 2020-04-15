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
        None: saves the datasets to a new file in their respective folders SSL_TS/ and SSL_RP/
    """
    T_pos = int(T_pos)
    T_neg = int(T_neg)

    onlyfiles = [f for f in listdir(file_folder) if isfile(join(file_folder, f))]

    SAMPLE_TIME = 30
    CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']

    for data_file in onlyfiles:
        path_file = file_folder + '/' + data_file
        print(data_file)
        data = PPD2(path_file, SAMPLE_TIME, CHANNELS)


        # If folder doesn't exist, then create it.
        if not os.path.isdir("SSL_TS"):
            os.makedirs("SSL_TS")
        if not os.path.isdir("SSL_RP/RP_dataset") or not os.path.isdir("SSL_TS/TS_labels"):
            os.makedirs("SSL_TS/TS_dataset")
            os.makedirs("SSL_TS/TS_labels")
        # temporal_shuffling(data, T_pos, T_neg):
        TS_dataset, TS_labels = temporal_shuffling(data, T_pos, T_neg)
        # Save Dataset
        np.save('SSL_TS/TS_dataset/' + str(data_file), TS_dataset)
        np.save('SSL_TS/TS_labels/' + str(data_file), TS_labels)

        # If folder doesn't exist, then create it.
        if not os.path.isdir("SSL_RP"):
            os.makedirs("SSL_RP")
        if not os.path.isdir("SSL_RP/RP_dataset") or not os.path.isdir("SSL_RP/RP_labels"):
            os.makedirs("SSL_RP/RP_dataset")
            os.makedirs("SSL_RP/RP_labels")
        ### relative_positioning(data, T_pos, T_neg):
        RP_dataset, RP_labels = relative_positioning(data, 3, 3)
        # Save Dataset
        np.save('SSL_RP/RP_dataset/' + str(data_file), RP_dataset)
        np.save('SSL_RP/RP_labels/' + str(data_file), RP_labels)

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    create(*sys.argv[1:])

### AAV :)
