# -*- coding: utf-8 -*-


import torch.nn as nn

from utils.nn import Maxout

# model parameters
nz = 100  # number of visual ujnis
keep_prob_default = 0.5
keep_prob_h0 = 0.8
scale_factor_default = 2
scale_factor_h0 = 1.25


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.L1 = nn.Sequential(
            nn.Linear(nz, 1200),
            nn.ReLU()
        )
        self.L2 = nn.Sequential(
            nn.Linear(1200, 1200),
            nn.ReLU()
        )
        self.L3 = nn.Sequential(
            nn.Linear(1200, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.L1 = nn.Sequential(
            nn.Dropout(1- keep_prob_h0),
            Maxout(784, 240, 5),
        )
        self.L2 = nn.Sequential(
            nn.Dropout(1 - keep_prob_default),
            Maxout(240, 240, 5),
        )
        self.L3 = nn.Sequential(
            nn.Dropout(1 - keep_prob_default),
            nn.Linear(240, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        return x
