from sklearn.metrics import balanced_accuracy_score, accuracy_score
import torch
from torch import nn
import numpy as np
from .train_helpers import normalize
from torch.utils import data

def get_test_results(model, test_loader):
	y_true = []
	y_pred = []
	model.eval()
	softmax = nn.Softmax()
	with torch.no_grad():
		for pair in test_loader:
			x, y = pair[0], pair[1]
			x = x.cuda().float().contiguous()
			y = y.cuda().long().contiguous()
			out = model(x)
			_, predicted = torch.max(softmax(out.data), 1)
			y_true.extend(list(y.cpu().numpy()))
			y_pred.extend(list(predicted.cpu().numpy()))
	return y_true, y_pred

def scores(model, epochs_test):
	X_test = normalize(epochs_test.get_data())
	y_test = epochs_test.events[:, 2] - 1
	test_dataset = data.TensorDataset(torch.tensor(X_test).unsqueeze(1), torch.tensor(y_test))
	test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False)
	
	y_true, y_pred = get_test_results(model, test_loader)
	acc_score = accuracy_score(y_true, y_pred)
	balanced_acc_score = balanced_accuracy_score(y_true, y_pred)
	print(f'Performance of the network on the {len(test_loader.dataset)} test images:')
	print(f'\tAccuracy: {100*acc_score:.2f}%')
	print(f'\tBalanced accuracy: {100*balanced_acc_score:.2f}%')
	return acc_score, balanced_acc_score
