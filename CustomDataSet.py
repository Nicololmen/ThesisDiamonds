# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:21:25 2022

@author: Milan
"""
import pandas as pd
import torch

class CustomDataSet():
  def __init__(self, filename, start, end, headers, drop_headers, model_type):
    self.data = pd.read_csv(filename, skiprows=start, nrows=end, names=headers)
    self.data = self.data.drop(columns=drop_headers)
    self.model_type = model_type
    
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    row=self.data.iloc[idx]
    label=row[0]
    features=row[1:642]
    features = torch.Tensor(features)

    #Add extra dim for Convolution network
    if(self.model_type):
      features = features[None, :]

    return features, int(label)