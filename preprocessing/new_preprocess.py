import os
import numpy as np
import mne
from mne import preprocessing
import sys


def preprocess(file):
    """ Runs the whole pipeline and returns NumPy data array"""
    SAMPLE_TIME = 30
    CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']
    
    raw = mne.io.read_raw_edf(file, preload=True)

    mne_eeg = remove_sleepEDF(raw, CHANNELS)
    
    mne_filtered = filter(mne_eeg, CHANNELS)
    
    epochs = divide_epochs(mne_filtered, SAMPLE_TIME)
    
    epochs = downsample(epochs, CHANNELS)

    epochs = epochs.get_data() # turns into NumPy Array

    f_epochs = normalization(epochs)

    #np.save(file[:file.index("-")], f_epochs)
    
    return f_epochs

def remove_sleepEDF(mne_raw, CHANNELS):
    """Extracts CHANNELS channels from MNE_RAW data.

    Args:
    raw - mne data strucutre of n number of recordings and t seconds each
    CHANNELS - channels wished to be extracted

    Returns:
    extracted - mne data structure with only specified channels
    """
    extracted = mne_raw.pick_channels(CHANNELS)
    return extracted

def filter(mne_eeg, chs):
    """Creates a 30 Hz 4th-order FIR lowpass filter that is applied to the CHS channels from the MNE_EEG data.

    Args:
        mne-eeg - mne data strucutre of n number of recordings and t seconds each

    Returns:
        filtered - mne data structure after the filter has been applied
    """
    
    filtered = mne_eeg.filter(l_freq=None,
            h_freq= 30,
            picks = chs,
            filter_length = "auto",
            method = "fir"
            )
    return filtered

def _create_events(raw, epoch_length):
    """Creates events at the right times split raw into epochs of length epoch_length seconds.

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
    E = epochs.pick_types(eeg=True, selection=chs)
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
    for i in range(epochs.shape[0]):
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