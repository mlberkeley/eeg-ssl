import torch
from torch import nn
from .feature_extractor import EEG_FeatureExtractor

class SupervisedBaseline(nn.Module):
	def __init__(self, C, T, n_classes, loss_weights, k=50, m=13, dropout_prob=0.5, n_spatial_filters=8):
		super().__init__()
		self.feature_extractor = EEG_FeatureExtractor(C, T, k, m, dropout_prob, n_classes, n_spatial_filters)
		self.loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

	def forward(self, x):
		out = self.feature_extractor(x)
		return out

	def loss(self, x, y_true):
		out = self(x)
		return self.loss_fn(out, y_true)