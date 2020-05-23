import torch
from torch import optim
from torch.utils import data
import numpy as np
from .train_helpers import normalize, get_loss_weights, load_losses, save_losses

from .models import SupervisedBaseline
import os.path as op
import os

root = op.dirname(__file__)
saved_models_dir = op.join(root, 'saved_models')

def train_supervised_baseline(epochs_train, epochs_test, n_epochs=20, lr=1e-3, batch_size=256, load_last_saved_model=False):
	X_train = normalize(epochs_train.get_data())
	
	y_train = epochs_train.events[:, 2] - 1 # start at 0

	X_test = normalize(epochs_test.get_data())
	y_test = epochs_test.events[:, 2] - 1
	
	n_classes = y_train.max() - y_train.min() + 1
	C = X_train.shape[1] # num channels
	T = X_train.shape[2] # window length
	loss_weights = get_loss_weights(epochs_train)
	model = SupervisedBaseline(C, T, n_classes, loss_weights).cuda()
	if load_last_saved_model:
		model.load_state_dict(torch.load(op.join(root, 'saved_models', 'supervised_baseline_model.pt')))

	train_dataset = data.TensorDataset(torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train))
	train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	test_dataset = data.TensorDataset(torch.tensor(X_test).unsqueeze(1), torch.tensor(y_test))
	test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

	new_train_losses, new_test_losses = _train_epochs(model, train_loader, test_loader, 
																				 dict(epochs=n_epochs, lr=lr))

	if load_last_saved_model:
		train_losses, test_losses = load_losses(saved_models_dir, 'supervised_baseline')
	else:
		train_losses = []
		test_losses = []
	
	train_losses.extend(new_train_losses)
	test_losses.extend(new_test_losses)

	save_losses(train_losses, test_losses, saved_models_dir, 'supervised_baseline')

	return train_losses, test_losses, model

def _train_epochs(model, train_loader, test_loader, train_args):
	epochs, lr = train_args['epochs'], train_args['lr']
	optimizer = optim.Adam(model.parameters(), lr=lr)
	if not os.path.exists(saved_models_dir):
		os.makedirs(saved_models_dir)
	
	train_losses = []
	test_losses = [_eval_loss(model, test_loader)]
	for epoch in range(1, epochs+1):
		model.train()
		train_losses.extend(_train(model, train_loader, optimizer, epoch))
		test_loss = _eval_loss(model, test_loader)
		test_losses.append(test_loss)
		print(f'Epoch {epoch}, Test loss {test_loss:.4f}')
		
		# save model every 10 epochs
		if epoch % 10 == 0:
			torch.save(model.state_dict(), op.join(root, 'saved_models', 'supervised_baseline_model_epoch{}.pt'.format(epoch)))
	torch.save(model.state_dict(), op.join(root, 'saved_models', 'supervised_baseline_model.pt'))
	return train_losses, test_losses

def _train(model, train_loader, optimizer, epoch):
	model.train()
	
	train_losses = []
	for pair in train_loader:
		x, y = pair[0], pair[1]
		x = x.cuda().float().contiguous()
		y = y.cuda().long().contiguous()
		loss = model.loss(x, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_losses.append(loss.item())
	return train_losses

def _eval_loss(model, data_loader):
	model.eval()
	total_loss = 0
	with torch.no_grad():
		for pair in data_loader:
			x, y = pair[0], pair[1]
			x = x.cuda().float().contiguous()
			y = y.cuda().long().contiguous()
			loss = model.loss(x, y)
			total_loss += loss * x.shape[0]
		avg_loss = total_loss / len(data_loader.dataset)

	return avg_loss.item()

