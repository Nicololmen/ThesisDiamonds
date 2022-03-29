# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:21:25 2022

@author: Milan
"""
import pandas as pd
import torch

class CustomDataSet():
  def __init__(self, filename, start, end, headers, drop_headers):
    self.data=pd.read_csv(filename, skiprows=start, nrows=end, names=headers)
    self.data=self.data.drop(columns=drop_headers)
    
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    row=self.data.iloc[idx]
    label=row[0]
    features=row[1:642]
    
    features=torch.Tensor(features)
    return features, int(label)