import pandas as pd
import torch
import cv2
import numpy as np
import os


class HybrideDataSet():
    def __init__(self, filename, start, end, headers, drop_headers):
        self.data = pd.read_csv(filename, skiprows=start, nrows=end, names=headers)
        self.data = self.data.drop(columns=drop_headers)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Get label
        label = row[6]

        # get spectral data
        features = row[7:648]

        # get wp, wr, int time, clarity fluo
        wp = row[1:2]
        wr = row[2:3]
        integrationTime = row[3:4] / 100000
        clarity = row[4]
        fluo = row[5]
        if (fluo == 'NON'):
            f = np.array([1, 0, 0, 0])
        elif (fluo == 'FNT'):
            f = np.array([0, 1, 0, 0])
        elif (fluo == 'MED'):
            f = np.array([0, 0, 1, 0])
        elif (fluo == 'STG'):
            f = np.array([0, 0, 0, 1])

        if (clarity == 'FL'):
            c = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif (clarity == 'IF'):
            c = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif (clarity == 'VVS1'):
            c = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif (clarity == 'VVS2'):
            c = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif (clarity == 'VS1'):
            c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif (clarity == 'VS2'):
            c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif (clarity == 'SI1'):
            c = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif (clarity == 'SI2'):
            c = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif (clarity == 'SI3'):
            c = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif (clarity == 'I1'):
            c = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif (clarity == 'I2'):
            c = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif (clarity == 'I3'):
            c = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        #wp = pd.concat([wp, wr])
        wp = pd.concat([wp, integrationTime]) 
        wp = np.array(wp, dtype=np.float32)
        wp = np.concatenate([wp, c])
        wp = np.concatenate([wp, f])

        # make tensors
        features = torch.Tensor(features)
        features2 = torch.from_numpy(wp.astype(np.float32))


        return features, features2, int(label)