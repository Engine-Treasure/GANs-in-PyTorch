# -*- coding: utf-8 -*-


import torch.nn as nn

from utils.nn import Maxout

from utils import initialize_weights

# model parameters
nz = 74  # number of visual units


# Network architecture is exactly same as in InfoGAN
# Refer to pytorch-generative-model-collections
# https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/GAN.py
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.L1 = nn.Sequential(
            nn.Linear(nz, 1024),
            nn.BatchNorm1d(1024),  # num_features - from an expected input of size batch_size x num_feafures
            nn.ReLU()
        )
        self.L2 = nn.Sequential(
            nn.Linear(1024, 128 * 7 * 7),  # 128 will be used as channel size
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU()
        )
        self.L3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # in channels, out channels, kernel size, stride, padding
            nn.BatchNorm2d(64),  # batch_size x `num_features` x height x width
            nn.ReLU(),
        )
        self.L4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        x = x.view(-1, 128, 7, 7)  # reshape to batch_size x channel_size x height x width
        x = self.L3(x)
        x = self.L4(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.L1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),  # in channels, out channels, kernel size, stride, padding
            nn.LeakyReLU(0.1)
        )
        self.L2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        self.L3 = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1)
        )
        self.L4 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        x = x.view(-1, 128 * 7 * 7)  # reshape as batch_size x 128*7*7
        x = self.L3(x)
        x = self.L4(x)

        return x
