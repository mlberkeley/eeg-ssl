# eeg-ssl
Self-supervised learning for EEG


## Pre-Processing_Dataset_2.py (Alfredo)
Input: E’ - mne data structure. <p>
Output: E - Dataset of epochs of 30 seconds of filtered and normalized data
  
#### Divide into 30s samples
Input: E’ - a mne data structure of n number of recordings and t seconds each. <p>
Output: E - dataset of epochs e of 30 seconds but temporally ordered. 
        #epochs = |E’| = (n * t) / 30

#### Downsample to 128 Hz and 3 channels 	[for MASS = 256 Hz]
Input: E’  - a mne data structure sampled at a rate r’ > 128 Hz <p>
Output: E’ - a mne data structure sampled at a rate r of 128 Hz. 
  
#### Normalization by sample 
Input: E’ - dataset (ds) of mne epoch objects eE’
Output: E  ds of mne epoch objects normalized s.t <p>
        mean(e)= 0, var(e) = 1 eE <p>

#### Return E - Dataset of labeled e 
