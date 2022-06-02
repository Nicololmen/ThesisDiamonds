import torch.nn as nn
from torchsummary import summary
import torch


class HybrideNet(nn.Module):
    def __init__(self, architecture_lijst):
        super(HybrideNet, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(641, 16), nn.ReLU())

        architecture_lijst = architecture_lijst.split('-')
        layers = []
        for index, (layer_in, layer_out) in enumerate(zip(architecture_lijst[0:], architecture_lijst[1:])):
            layers.append(nn.Linear(int(layer_in), int(layer_out)))
            if (index < len(architecture_lijst) - 2):
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)
        print(summary(self.mlp, (0, 0, 641)))
        print(summary(self.model, (0, 0, 34)))

    def forward(self, features, z):

        
        y = self.mlp(features)
        f = torch.cat((y, z), 1)
        return self.model(f)

