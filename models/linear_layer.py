import torch
from torch import nn

class SSL_Linear(nn.Module):
	def __init__(self, model, loss_weights):
		super().__init__()
		self.model = model
		self.model.requires_grad = False
		self.linear = nn.Linear(100, 5)
		self.loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

	def forward(self, x):
		with torch.no_grad():
			features = self.model.feature_extractor(x)
		out = self.linear(features)
		return out

	def loss(self, x, y_true):
		out = self(x)
		return self.loss_fn(out, y_true)