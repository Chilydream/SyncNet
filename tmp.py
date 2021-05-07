import pickle
import numpy as np
import torch
import torchsnooper
import torch.nn.functional as F
import torch.nn as nn


class LossScale(nn.Module):
	def __init__(self, init_w=10.0, init_b=-5.0):
		super(LossScale, self).__init__()

		self.wI = nn.Parameter(torch.tensor(init_w))
		self.bI = nn.Parameter(torch.tensor(init_b))

		self.wC = nn.Parameter(torch.tensor(init_w))
		self.bC = nn.Parameter(torch.tensor(init_b))


a = LossScale()
b = LossScale()
for pname, p in a.named_parameters():
	print(pname, p)
