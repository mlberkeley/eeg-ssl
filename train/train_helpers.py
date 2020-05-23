import numpy as np
import torch
import os.path as op

def normalize(x):
	x_normalized = (x - x.mean(2).reshape(x.shape[0],x.shape[1],1))/(x.std(2).reshape(x.shape[0],x.shape[1],1))
	return x_normalized

def normalize_one(x):
	x_normalized = (x - x.mean(1).reshape(x.shape[0],1))/(x.std(1).reshape(x.shape[0],1))
	return x_normalized

def get_loss_weights(epochs_train):
	y_train = epochs_train.events[:, 2] - 1 # start at 0
	counts = np.bincount(y_train)
	weights = len(y_train) / (counts * len(counts))
	print("Class weights", weights)
	return torch.from_numpy(weights).cuda().float()

def load_losses(saved_models_dir, name):
	with open(op.join(saved_models_dir, name + '_train_losses.npy'), 'rb') as f:
		train_losses = list(np.load(f))
	with open(op.join(saved_models_dir, name + '_test_losses.npy'), 'rb') as f:
		test_losses = list(np.load(f))
	return train_losses, test_losses

def save_losses(train_losses, test_losses, saved_models_dir, name):
	with open(op.join(saved_models_dir, name + '_train_losses.npy'), 'wb') as f:
		np.save(f, train_losses)
	with open(op.join(saved_models_dir, name + '_test_losses.npy'), 'wb') as f:
		np.save(f, test_losses)