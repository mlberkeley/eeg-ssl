from preprocessing.preprocessing import preprocess
from ssl.SSL_create_TS_RP import create
import sys


def eeg_ssl(data_folder, T_pos_RP, T_neg_RP, T_pos_TS, T_neg_TS):
     """ Divides the mne dataset into many samples of length e_len seconds.

     Args:
        data_folder: folder containing EEG .edf files
        T_pos_RP: an integer representing the positive limit for relative positioning.
        T_neg_RP: an integer representing the negative limit for relative positioning.
        T_pos_TS: an integer representing the positive limit for temporal shuffling.
        T_neg_TS: an integer representing the negative limit for temporal shuffling.


     Returns:
        
     """
     preprocessed = preprocess(data_folder)
     create(preprocessed, T_pos_RP, T_neg_RP, T_pos_TS, T_neg_TS)

if __name__ == '__main__':
    data_folder = sys.argv[1]
    T_pos_RP = sys.argv[2]
    T_neg_RP = sys.argv[3]
    T_pos_TS = sys.argv[2]
    T_neg_TS = sys.argv[3]
    eeg_ssl(data_folder, T_pos_RP, T_neg_RP, T_pos_TS, T_neg_TS)

# AC :)