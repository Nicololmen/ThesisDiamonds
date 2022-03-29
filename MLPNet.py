# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:24:47 2022

@author: Milan
"""


import torch.nn as nn
from torchsummary import summary


class Net(nn.Module):
  def __init__(self, architecture_lijst):
    super(Net, self).__init__()
    architecture_lijst = architecture_lijst.split('-')
    layers = []
    for index, (layer_in, layer_out) in enumerate(zip(architecture_lijst[0:], architecture_lijst[1:])):
      layers.append(nn.Linear(int(layer_in), int(layer_out)))
      if (index < len(architecture_lijst) - 2):
        layers.append(nn.ReLU())
    self.model = nn.Sequential(*layers)
    print(summary(self.model, (0, 0, 641)))

  def forward(self, input):
    return self.model(input)

