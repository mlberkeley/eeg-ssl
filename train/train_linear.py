import numpy as np
import torch
from torch import optim
from torch.utils import data

from .train_helpers import normalize, get_loss_weights
from .models import SSL_Linear

from .train_supervised_baseline import _train, _eval_loss

def train_linear(epochs_train, epochs_test, model, n_epochs=5, lr=1e-3, batch_size=256):
  X_train = normalize(epochs_train.get_data())
  y_train = epochs_train.events[:, 2] - 1 # start at 0

  X_test = normalize(epochs_test.get_data())
  y_test = epochs_test.events[:, 2] - 1
	
  loss_weights = get_loss_weights(epochs_train)

  linear_model = SSL_Linear(model, loss_weights).cuda()

  train_dataset = data.TensorDataset(torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train))
  train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  test_dataset = data.TensorDataset(torch.tensor(X_test).unsqueeze(1), torch.tensor(y_test))
  test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  train_losses, test_losses = _train_epochs(linear_model, train_loader, test_loader, 
                                         dict(epochs=n_epochs, lr=lr))

  return train_losses, test_losses, linear_model

def _train_epochs(model, train_loader, test_loader, train_args):
	epochs, lr = train_args['epochs'], train_args['lr']
	optimizer = optim.Adam(model.parameters(), lr=lr)
	
	train_losses = []
	test_losses = [_eval_loss(model, test_loader)]
	for epoch in range(1, epochs+1):
		model.train()
		train_losses.extend(_train(model, train_loader, optimizer, epoch))
		test_loss = _eval_loss(model, test_loader)
		test_losses.append(test_loss)
		print(f'Epoch {epoch}, Test loss {test_loss:.4f}')
	return train_losses, test_losses