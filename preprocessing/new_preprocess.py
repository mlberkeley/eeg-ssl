import os
import numpy as np
import mne
from mne import preprocessing
import sys
import time

def preprocess(f):
    """ Runs the whole pipeline and returns NumPy data array"""
    epoch_length = 30 # s
    CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']
    
    raw = mne.io.read_raw_edf(f, preload=True)
    mne_eeg = remove_sleepEDF(raw, CHANNELS)
    mne_filtered = filter_eeg(mne_eeg, CHANNELS)
    epochs = divide_epochs(mne_filtered, epoch_length)
    
    # epochs = downsample(epochs, CHANNELS) [it's already at 100 Hz]

    f_epochs = normalization(epochs) # should update this

    #np.save(file[:file.index("-")], f_epochs)
    
    return f_epochs

def remove_sleepEDF(mne_raw, channels):
    """Extracts CHANNELS channels from MNE_RAW data.

    Args:
    raw - mne data strucutre of n number of recordings and t seconds each
    channels - channels wished to be extracted

    Returns:
    extracted - mne data structure with only specified channels
    """
    extracted = mne_raw.pick_channels(channels)
    return extracted

def filter_eeg(mne_eeg, channels):
    """Creates a 30 Hz 4th-order FIR lowpass filter that is applied to the channels channels from the MNE_EEG data.

    Args:
        mne-eeg - mne data strucutre of n number of recordings and t seconds each

    Returns:
        filtered - mne data structure after the filter has been applied
    """
    
    filtered = mne_eeg.filter(l_freq=None,
            h_freq= 30,
            picks = channels,
            filter_length = "auto",
            method = "fir",
            verbose='ERROR'
            )
    return filtered

def divide_epochs(raw, epoch_length):
    """ Divides the mne dataset into many samples of length epoch_length seconds.

    Args:
        E: mne data structure
        epoch_length: (int seconds) length of each sample

    Returns:
        epochs: mne data structure of (experiment length * users) / epoch_length
    """

    raw_np = raw.get_data()
    s_freq = raw.info['sfreq']
    n_channels, n_time_points = raw_np.shape[0], raw_np.shape[1]

    # make n_time_points a multiple of epoch_length*s_freq
    chopped_n_time_points = n_time_points - (n_time_points % int(epoch_length*s_freq))
    raw_np = raw_np[:,:chopped_n_time_points]

    return raw_np.reshape(n_channels, epoch_length*s_freq)

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
    E = epochs.pick_types(eeg=True, selection=chs, verbose='ERROR')
    E = E.resample(Hz, npad='auto')
    return E

def _normalize(epoch):
    """ A helper method for the normalization method.

        Args:
            epochs: mne data structure sampled at a rate r’ > 128 Hz

        Returns
            result: a normalized epoch
    """
    result = (epoch - epoch.mean(axis=0)) / (np.sqrt(epoch.var(axis=0)))
    return result

def normalization(epochs):
    """ Normalizes each epoch e s.t mean(e) = 0 and var(e) = 1

        Args:
            epochs - Numpy structure of epochs

        Returns:
            epochs_n - mne data structure of normalized epochs (mean=0, var=1)
    """
    for i in range(epochs.shape[0]): # TODO could switch to a 1-line numpy matrix operation
        for j in range(epochs.shape[1]):
            epochs[i,j,:] = _normalize(epochs[i,j,:])

    return epochs

def save(f_epochs, name, output_folder):
    """ Saves each epoch as a file

    Args:
    f_epochs - Numpy structure of epochs

    name - file name based on its original mne file name

    output_folder - folder name where data should be saved
    """
    np.save(output_folder + "/epoch_{num}.npy".format(num = name), f_epochs)

if __name__ == '__main__':
    data_folder = sys.argv[1]
    preprocess(data_folder)

            
### AC + AAV :)            