# Self-Supervised Representation Learning From Electroencephalography Signals

[![ML@B](https://github.com/rhotter/eeg-ssl/blob/master/images/mlab_logo.png)](https://ml.berkeley.edu)

This implementation is based on the [paper](https://arxiv.org/pdf/1911.05419.pdf) by Banville et al.

--add more description about program here
# Directories
* `experiments` : Improvement experiments folder


# Installation and tutorial
## Dependencies

## Installation
```sh
$ python eeg_ssl.py data_folder T_pos_RP T_neg_RP T_pos_TS T_neg_TS
```
- data_folder: a folder containing EEG .edf files
- T_pos_RP: an integer representing the positive limit for relative positioning.
- T_neg_RP: an integer representing the negative limit for relative positioning.
- T_pos_TS: an integer representing the positive limit for temporal shuffling.
- T_neg_TS: an integer representing the negative limit for temporal shuffling.



License
----

MIT
