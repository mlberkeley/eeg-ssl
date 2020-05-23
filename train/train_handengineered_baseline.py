from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import balanced_accuracy_score
from mne.time_frequency import psd_welch

import numpy as np

def hand_engineered_baseline(epochs_train, epochs_test):
	pipe = make_pipeline(FunctionTransformer(_eeg_power_band, validate=False),
										RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
	# Train
	print("Training with balanced class weights")
	y_train = epochs_train.events[:, 2]
	pipe.fit(epochs_train, y_train)

	# Test
	y_pred = pipe.predict(epochs_test)

	# Assess the results
	y_test = epochs_test.events[:, 2]
	acc = accuracy_score(y_test, y_pred)
	balanced_acc = balanced_accuracy_score(y_pred, y_test)

	print(f'Accuracy: {100*acc:.2f}%')
	print(f'Balanced accuracy: {100*balanced_acc:.2f}%')

def _eeg_power_band(epochs):
		"""EEG relative power band feature extraction.

		This function takes an ``mne.Epochs`` object and creates EEG features based
		on relative power in specific frequency bands that are compatible with
		scikit-learn.

		Parameters
		----------
		epochs : Epochs
				The data.

		Returns
		-------
		X : numpy array of shape [n_samples, 5]
				Transformed data.
		"""
		# specific frequency bands
		FREQ_BANDS = {"delta": [0.5, 4.5],
									"theta": [4.5, 8.5],
									"alpha": [8.5, 11.5],
									"sigma": [11.5, 15.5],
									"beta": [15.5, 30]}

		psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=30.)
		# Normalize the PSDs
		psds /= np.sum(psds, axis=-1, keepdims=True)

		X = []
		for fmin, fmax in FREQ_BANDS.values():
			psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
			X.append(psds_band.reshape(len(psds), -1))

		return np.concatenate(X, axis=1)