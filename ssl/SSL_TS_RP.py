import os
import numpy as np
import pandas as pd
import mne
import pickle
import random
from mne import preprocessing


def temporal_shuffling(epochs, T_pos, T_neg):
    """ Builds a self-supervised (temporal shuffling) dataset of epochs

    Args:
        E - Numpy datset of time-series arrays
        T_pos - positive context to sample from
        T_neg - negative context to sample from

    Output:
        TS_dataset - Temporal Shuffling Dataset of dimensions (L, 4, s, c)
            L - # of samples = # of user * # of epochs per user * 6
            3 - sample1 + sample2 + sample3
            s - # of eeg channels in each sample
            c - Samples per channel = 30s * 128Hz
        TS_labels - Temporal Shuffling labels of dimensions (L, 1)
            for each y = {1: if sample1 < sample2 < sample3 and -1: otherwise}
    """
    total_samples = epochs.shape[0] * 6
    TS_dataset = np.empty((total_samples, 3, epochs.shape[1], 3867))
    TS_labels = np.empty((total_samples, 1))
    counter = 0
   
    for idx, sample1 in enumerate(epochs):
        for _ in range(3): # T_pos loop
            sample2_index = np.random.randint(max(idx-T_pos, 0), min(idx+T_pos, epochs.shape[0]-1))
            while sample2_index == idx: # should not be the same
                sample2_index = np.random.randint(max(idx-T_pos, 0), min(idx+T_pos, epochs.shape[0]-1))
            sample2 = epochs[sample2_index]

            if idx-T_neg <= 0: # T_neg if (corners)
                sample3_index = np.random.randint(idx+T_neg, epochs.shape[0])
            elif idx+T_neg >= epochs.shape[0]: # take care of low == high
                sample3_index = np.random.randint(0, idx-T_neg)
            else:
                sample3_index_1 = np.random.randint(idx+T_neg, epochs.shape[0])
                sample3_index_2 = np.random.randint(0, idx-T_neg)
                sample3_index = list([sample3_index_1, sample3_index_2])[int(random.uniform(0,1))]
            sample3 = epochs[sample3_index]

            if idx < sample2_index and sample2_index < sample3_index:
                y = 1
            else:
                y = -1

            TS_sample = np.array([sample1, sample2, sample3])
            TS_dataset[counter] = TS_sample
            TS_labels[counter] = y
            counter += 1

        for _ in range(3): # T_neg loop
            sample2_index = np.random.randint(max(idx-T_pos, 0), min(idx+T_pos, epochs.shape[0]-1))
            while sample2_index == idx: # should not be the same
                sample2_index = np.random.randint(max(idx-T_pos, 0), min(idx+T_pos, epochs.shape[0]-1))
            sample2 = epochs[sample2_index]

            sample3_index = np.random.randint(max(idx-T_pos, 0), min(idx+T_pos, epochs.shape[0]-1))
            while sample2_index == sample3_index or sample3_index == idx: # should not be the same
                sample3_index = np.random.randint(max(idx-T_pos, 0), min(idx+T_pos, epochs.shape[0]-1))
            sample3 = epochs[sample3_index]

            if idx < sample2_index and sample2_index < sample3_index:
                y = 1
            else:
                y = -1

            TS_sample = np.array([sample1, sample2, sample3])
            TS_dataset[counter] = TS_sample
            TS_labels[counter] = y
            counter += 1

    return TS_dataset, TS_labels


def relative_positioning(epochs, T_pos, T_neg):
    """ Builds a self-supervised (relative positioning) dataset of epochs

    Args:
        E - Numpy datset of time-series arrays
        T_pos - positive context to sample from
        T_neg - negative context to sample from

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
    RP_dataset = np.empty((total_samples, 2, epochs.shape[1], 3867))
    RP_labels = np.empty((total_samples, 1))
    counter = 0

    for idx, sample1 in enumerate(epochs):
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

    return RP_dataset, RP_labels

### AAV :)
