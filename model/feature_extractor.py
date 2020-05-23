import torch
from torch import nn
import numpy as np

class EEG_FeatureExtractor(nn.Module):
    # based on "A deep learning architecture for temporal sleep stage
    # classification using multivariate and multimodal time series"

  def __init__(self, C, T, k=50, m=13, dropout_prob=0.5, embedding_dim=100, n_spatial_filters=8):
    """
    C: number of EEG channels
    T: number of timepoints in a window
    k: length of spatial filters (i.e. how much you look in time)
    m: maxpool size
    n_spatial_filters: number of spatial filters
    embedding_dim: embedding dimension (D)
    """
    # input is (1, C, T) <-- notation (channels, dim1, dim2) is different than paper (dim1, dim2, channels)
    super().__init__()
    self.depthwise_conv = nn.Conv2d(in_channels=1, out_channels=C, kernel_size=(C,1))
    self.spatial_padding = nn.ReflectionPad2d((int(np.floor((k-1)/2)),int(np.ceil((k-1)/2)),0,0))
    self.spatialwise_conv1 = nn.Conv2d(in_channels=1, out_channels=n_spatial_filters, kernel_size=(1,k))
    self.spatialwise_conv2 = nn.Conv2d(in_channels=n_spatial_filters, out_channels=n_spatial_filters, kernel_size=(1,k))
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=(1,m), stride=(1,m))
    self.dropout = nn.Dropout(p=dropout_prob, inplace=True)
    self.linear = nn.Linear(n_spatial_filters * C * ((T // m) // m), embedding_dim)

  def forward(self, x):
    # input is (bs, 1, C, T)
    bs = x.shape[0]
    out = x
    out = self.depthwise_conv(out) # (bs, C, 1, T)
    out = out.permute(0,2,1,3) # (bs, 1, C, T)
    out = self.spatial_padding(out)
    out = self.spatialwise_conv1(out) # (bs, n_spatial_filters, C, T)
    out = self.relu(out)
    out = self.maxpool(out) # (bs, n_spatial_filters, C, T // m)
    out = self.spatial_padding(out)
    out = self.spatialwise_conv2(out) # (bs, n_spatial_filters, C, T // m)
    out = self.relu(out)
    out = self.maxpool(out) # (bs, n_spatial_filters, C, (T // m) // m)
    out = out.view(bs, -1) # (bs, n_spatial_filters * C * ((T // m) // m))
    out = self.dropout(out)
    out = self.linear(out) # (bs, embedding_dim)
    return out