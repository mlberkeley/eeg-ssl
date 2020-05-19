import os
import numpy as np
import sys
from os import listdir
from os.path import isfile, join

from ssl.SSL_TS_RP import temporal_shuffling, relative_positioning

def create(preprocessed, T_pos_RP, T_neg_RP, T_pos_TS, T_neg_TS):
    """ Runs the whole pipeline and saves Numpy datasets and labels.

    Args:
        preprocessed: The list of numpy arrays of epochs and their associated original file name
                      [epoch numpy, file name (string)]
        T_pos_RP: (time) sampling hyperparameter for po
        T_neg_RP: (time) sampling hyperparameter
        T_pos_TS: (time) sampling hyperparameter
        T_neg_TS: (time) sampling hyperparameter

    returns
        None: saves the datasets to a new file in their respective folders SSL_TS/ and SSL_RP/
    """
    T_pos_RP = int(T_pos_RP)
    T_neg_RP = int(T_neg_RP)
    T_pos_TS = int(T_pos_TS)
    T_neg_TS = int(T_neg_TS)

    for data in preprocessed:
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

        # If folder doesn't exist, then create it.
        if not os.path.isdir("SSL_RP"):
            os.makedirs("SSL_RP")
        if not os.path.isdir("SSL_RP/RP_dataset") or not os.path.isdir("SSL_RP/RP_labels"):
            os.makedirs("SSL_RP/RP_dataset")
            os.makedirs("SSL_RP/RP_labels")
        
        ### relative_positioning(data, T_pos, T_neg):
        RP_dataset, RP_labels = relative_positioning(epoch_data, T_pos_RP, T_neg_RP)
        
        # Save Dataset
        np.save('SSL_RP/RP_dataset/' + file_name, RP_dataset)
        np.save('SSL_RP/RP_labels/' + file_name, RP_labels)

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    preprocessed = sys.argv[1]
    T_pos_RP = sys.argv[2]
    T_neg_RP = sys.argv[3]
    T_pos_TS = sys.argv[4]
    T_neg_TS = sys.argv[5]
    create(preprocessed, T_pos_RP, T_neg_RP, T_pos_TS, T_neg_TS)

### AAV :)
