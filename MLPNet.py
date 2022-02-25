# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:24:47 2022

@author: Milan
"""


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(641, 16)
    self.fc2 = nn.Linear(16, 6)
        
  def forward(self, input):
    x = F.relu(self.fc1(input))
    x = self.fc2(x)
    return x

