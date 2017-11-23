# -*- coding: utf-8 -*-


import torch
import torch.nn as nn

from utils.nn import OneHotEncoder, Maxout

# model parameters
keep_prob = 0.5
nz = 100


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.L1z = nn.Sequential(
            nn.Linear(nz, 200),  # layer 1 applied to z
            nn.BatchNorm1d(200),
            nn.ReLU()
        )
        self.L1y = nn.Sequential(
            OneHotEncoder(10),
            nn.Linear(10, 1000),  # layer 1 applied to y, the additional info (condition)
            nn.BatchNorm1d(1000),
            nn.ReLU()
        )
        self.L2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1200, 784),
            nn.Sigmoid()
        )

        self.ohencoder = OneHotEncoder(10)

    def forward(self, z, y):
        z1 = self.L1z(z)
        y1 = self.L1y(y)
        x = torch.cat([z1, y1], dim=1)  # cat z1 and y1
        x = self.L2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.L1x = nn.Sequential(
            Maxout(784, 240, 5),
        )
        self.L1y = nn.Sequential(
            OneHotEncoder(10),
            Maxout(10, 50, 5),
        )
        self.L2 = nn.Sequential(
            nn.Dropout(p=0.5),
            Maxout(290, 240, 4),
        )
        self.L3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(240, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x1 = self.L1x(x)  # should be L1x, that's a mistake.
        y1 = self.L1y(y)
        x = torch.cat([x1, y1], dim=1)
        x = self.L2(x)
        x = self.L3(x)
        return x
