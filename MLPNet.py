# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:24:47 2022

@author: Milan
"""


import torch.nn as nn



class Net(nn.Module):
  def __init__(self, architecture_lijst):
    super(Net, self).__init__()
    architecture_lijst = architecture_lijst.split('-')
    layers = []
    for index, (layer_in, layer_out) in enumerate(zip(architecture_lijst[0:], architecture_lijst[1:])):
      print(layer_in, layer_out)
      layers.append(nn.Linear(int(layer_in), int(layer_out)))
      if (index != len(architecture_lijst) - 1):
        layers.append(nn.ReLU())
    self.model = nn.Sequential(*layers)

  def forward(self, input):
    return self.model(input)

