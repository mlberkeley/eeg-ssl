import torch
from torch import nn

class SleepStageEEG(nn.Module):
	def __init__(self, C, T, k, m):
		# input is (1, C, T) <-- notation (channels, dim1, dim2) is different than paper (dim1, dim2, channels)
		self.n_channels = C
		self.n_timepoints = T
		self.k = k # how much you look in time
		
		self.depthwise_conv = nn.Conv2d(in_channels=1, out_channels=C, kernel_size=(C,1))
		self.spatialwise_conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1,k))
		self.relu = nn.ReLU(inplace=True)
	
	def forward(self, x):
		# input is (1, C, T)
		out = x
		out = self.depthwise_conv(out) # (C, T, 1)
		out = out.permute(2,0,1) # (1, C, T)
		out = self.spatialwise_conv(out) # (8, C, T)
		out = self.relu(out)
		













		