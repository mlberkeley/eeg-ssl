import os
import numpy as np
import mne
from mne import preprocessing
import sys
import time

def print_time(f, *args):
  print("time for: " + f.__name__)
  start_time = time.time()
  return_val = f(*args)
  print(time.time() - start_time)
  return return_val

def preprocess(file):
    """ Runs the whole pipeline and returns NumPy data array"""
    epoch_length = 30 # s
    CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']
    
    def a():
        return mne.io.read_raw_edf(file, preload=True)

    raw = print_time(a)
    mne_eeg = print_time(remove_sleepEDF,raw, CHANNELS)
    mne_filtered = print_time(filter_eeg,mne_eeg, CHANNELS)
    epochs = print_time(divide_epochs,mne_filtered, epoch_length)
    
    # epochs = print_time(downsample(epochs, CHANNELS) [it's already at 100 Hz]

    epochs = print_time(epochs.get_data) # turns into NumPy Array

    f_epochs = print_time(normalization,epochs) # should update this

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
            verbose='WARNING'
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

    n_epochs = file_length // n_samp_in_epoch

    events = []
    for i_epoch in range(n_epochs):
        events.append([first_samp + i_epoch * n_samp_in_epoch, 0, 0])
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
    E = epochs.pick_types(eeg=True, selection=chs, verbose='WARNING')
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