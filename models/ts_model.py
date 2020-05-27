import torch
from torch import nn
from .feature_extractor import EEG_FeatureExtractor

class Temporal_Shuffling(nn.Module):
  def __init__(self, C, T, k=50, m=13, dropout_prob=0.5, embedding_dim=100, n_spatial_filters=8):
    super().__init__()
    self.feature_extractor = EEG_FeatureExtractor(C, T, k, m, dropout_prob, embedding_dim, n_spatial_filters).cuda()
    self.linear = nn.Linear(2*embedding_dim, 1)
    self.loss_fn = nn.BCEWithLogitsLoss()

  def forward(self, x):
    first_samples = x[:,0].unsqueeze(dim=1)
    second_samples = x[:,1].unsqueeze(dim=1)
    third_samples = x[:,2].unsqueeze(dim=1)

    h_1 = self.feature_extractor(first_samples) # (bs, z_dim)
    h_2 = self.feature_extractor(second_samples)
    h_3 = self.feature_extractor(third_samples)

    h_1_2 = torch.abs(h_1 - h_2)
    h_2_3 = torch.abs(h_2 - h_3)

    h_combined = torch.cat((h_1_2, h_2_3), dim=1)

    out = self.linear(h_combined)
    return out
  
  def loss(self, x, labels):
    out = self(x)
    return self.loss_fn(out, labels)