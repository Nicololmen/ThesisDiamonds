# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:23:38 2022

@author: Milan
"""

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv1d(1, 16, 5, stride=1)
    self.maxp1 = nn.MaxPool1d(2, stride=2)
    self.conv2 = nn.Conv1d(16, 32, 3, stride=1)
    self.maxp2 = nn.MaxPool1d(2, stride=2)
    self.conv3 = nn.Conv1d(32, 64, 3, stride=1)
    self.maxp3 = nn.MaxPool1d(2, stride=2)
    
    self.fc1 = nn.Linear(64 * 78, 64)
    self.fc2 = nn.Linear(64, 6)

    
    
        
  def forward(self, input):
    x = F.relu(self.conv1(input))
    x = self.maxp1(x)
    x = F.relu(self.conv2(x))
    x = self.maxp2(x)
    x = F.relu(self.conv3(x))
    x = self.maxp3(x)
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x= self.fc2(x)
    
    return x