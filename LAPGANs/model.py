# -*- coding: utf-8 -*-

import torch.nn as nn

import torchvision.datasets
import torchvision.datasets as dset

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(3 + 1, 64, 7, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 368, 7, (1, 4)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(368),
            nn.Dropout2d(0.5),
            nn.ConvTranspose2d(368, 128, 7, (1, 4)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(368),
            nn.
        )

    def forward(self, cz):