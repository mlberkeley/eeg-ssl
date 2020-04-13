import os
import numpy as np
import pandas as pd
import mne
from mne import preprocessing
import pickle
import random

def get_pickle_data():
    """This unpickles mne data file and returns a raw MNE data object """
    pickle_off = open('data1.pkl', 'rb')
    print(pickle_off)
    raw = pickle.load(pickle_off)[0]
    print(raw)
    pickle_off.close()
    return raw


### Toy data to work with
def get_fif_data(data_file):
    """takes in the data_file path and returns the raw MNE data object """
    raw = mne.io.read_raw_fif(data_file, preload=True)
    return raw


def _create_events(raw, epoch_length):
    """Create events at the right times split raw into epochs of length epoch_length seconds

    Args:
        raw - mne data strucutre of n number of recordings and t seconds each
        epoch_length - (seconds) the length of each outputting epoch.

    Returns:
        events - Numpy array of events  of epochs
    """
    file_length = raw.n_times
    first_samp = raw.first_samp
    sfreq = raw.info['sfreq']
    n_samp_in_epoch = int(epoch_length * sfreq)

    n_epochs = int(file_length // n_samp_in_epoch)

    events = []
    for i_epoch in range(n_epochs):
        events.append([first_samp + i_epoch * n_samp_in_epoch, int(0), int(0)])
    events = np.array(events)
    return events


def divide_epochs(raw, e_len):
    """ Divides the mne dataset into many samples of length e_len seconds.

    Args:
        E: mne data structure
        e_len: (int seconds) length of each sample

    Returns:
        epochs: mne data structure of (experiment length * users) / e_len
    """
    if raw.times[-1] >= e_len:
        events = _create_events(raw, e_len)
    print(events)
    epochs = mne.Epochs(raw, events=events, tmax=e_len, preload=True)
    return epochs


def downsample(epochs, chs, Hz=128):
    """ Downsample the EEG epoch to Hz=128 Hz and to only
        include the channels in ch.

        Args:
            epochs: mne data structure sampled at a rate r’ > 128 Hz
            chs: list of the channels to keep
            Hz: Hz to downsample to (default 128 Hz)
        Returns
            E: a mne data structure sampled at a rate r of 128 Hz.
    """
    # ch_id = [epochs.ch_names.index(ch) for ch in chs]
    E = epochs.pick_types(eeg=True, selection=chs)
    E = E.resample(Hz, npad='auto')
    return E


def _normalize(epoch):
    return (epoch - epoch.mean()) / epoch.var()

def normalization(epochs):
    """ Normalizes each epoch e s.t mean(e)=mean and var(e)=variance

        Args:
            epochs - Numpy structure of epochs

        Returns:
            epochs_n - mne data structure of normalized epochs (mean=0, var=1)
    """
    for i in range(epochs.shape[0]):
        epochs[i,:,:] = _normalize(epochs[i,:,:])

    return epochs


def PPD2(data_file, SAMPLE_TIME, CHANNELS):
    SAMPLE_TIME = 30
    CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']

    raw = get_pickle_data(data_file)

    epochs = divide_epochs(raw, SAMPLE_TIME)

    epochs = downsample(epochs, CHANNELS)

    epochs = epochs.get_data() # turns into NumPy Array

    f_epochs = normalization(epochs)

    return f_epochs


### AAV :)
