import torch
from torch import nn
from .feature_extractor import EEG_FeatureExtractor
import IPython
e = IPython.embed

class Relative_Positioning(nn.Module):
  def __init__(self, C, T, k=50, m=13, dropout_prob=0.5, embedding_dim=100, n_spatial_filters=8):
    super().__init__()
    self.feature_extractor = EEG_FeatureExtractor(C, T, k, m, dropout_prob, embedding_dim, n_spatial_filters).cuda()
    self.linear = nn.Linear(embedding_dim, 1)
    self.loss_fn = nn.BCEWithLogitsLoss()

  def forward(self, x):
    first_samples = x[:,0].unsqueeze(dim=1)
    second_samples = x[:,1].unsqueeze(dim=1)

    h_first = self.feature_extractor(first_samples)
    h_second = self.feature_extractor(second_samples)

    h_combined = torch.abs(h_first - h_second)

    out = self.linear(h_combined)
    return out
  
  def loss(self, x, labels):
    out = self(x)
    e()
    return self.loss_fn(out, labels)