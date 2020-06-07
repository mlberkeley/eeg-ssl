# Open Sourced Self-Supervised Representation Learning From Electroencephalography Signals

[![ML@B](https://github.com/rhotter/eeg-ssl/blob/master/images/mlab_logo.png)](https://ml.berkeley.edu)

This implementation is based on the [paper](https://arxiv.org/pdf/1911.05419.pdf) by Banville et al.

--add more description about program here
# Directories
* `experiments` : Improvement experiments folder
* `images` : Folder for images for README and results
* `model` : Folder for model scripts
* `preprocessing` : Folder for preprocessing scripts
* `ssl` : Folder for SSL scripts

# Installation and tutorial


## Installation
```sh
$ python eeg_ssl.py data_folder T_pos_RP T_neg_RP T_pos_TS T_neg_TS
```
Inputs
- data_folder: a folder containing EEG .edf files
- T_pos_RP: an integer representing the positive limit for relative positioning.
- T_neg_RP: an integer representing the negative limit for relative positioning.
- T_pos_TS: an integer representing the positive limit for temporal shuffling.
- T_neg_TS: an integer representing the negative limit for temporal shuffling.

Outputs
- RP_dataset: pairs of 30 second normalized EEG time windows
- RP_labels: 
  - +1 if the distance between the two windows is T_pos_RP
  - -1 if the distance between the two windows is T_neg_RP
- TS_dataset:  triples of 30 second normalized EEG time windows
- TS_labels:
  - +1 if the distance between the two windows is T_pos_TS
  - -1 if the distance between the two windows is T_neg_TS



License
----

MIT
