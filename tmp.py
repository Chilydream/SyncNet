import pickle
import numpy as np
import torch
import torchsnooper
import torch.nn.functional as F
import torch.nn as nn
from pytorch_metric_learning import losses

from DatasetLoader import MyDataLoader

loss_func = losses.NTXentLoss()
valLoader = MyDataLoader("data/test.txt", 30, True)
