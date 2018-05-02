# -*- coding: utf-8 -*-

import torch.nn as nn


class Generator_x2z(nn.Module):
    def __init__(self):
        super(Generator_x2z, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 1),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(128),
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(256),
        )
        self.Conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(512),
        )
        self.Conv6 = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(512),
        )
        self.Conv7 = nn.Sequential(
            nn.Conv2d(512, 256 * 2, 1, 1),  # as the original code shows, 256 * 2
        )

    def forward(self, x):
        z = self.Conv1(x)
        z = self.Conv2(z)
        z = self.Conv3(z)
        z = self.Conv4(z)
        z = self.Conv5(z)
        z = self.Conv6(z)
        z = self.Conv7(z)
        return z


class Generator_z2x(nn.Module):
    def __init__(self):
        super(Generator_z2x, self).__init__()
        self.TConv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 1),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(256),
        )
        self.TConv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(128),
        )
        self.TConv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.TConv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        self.TConv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 5, 1),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        self.Conv6 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        self.Conv7 = nn.Sequential(
            nn.Conv2d(32, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.TConv1(z.view(-1, 256, 1, 1))
        x = self.TConv2(x)
        x = self.TConv3(x)
        x = self.TConv4(x)
        x = self.TConv5(x)
        x = self.Conv6(x)
        x = self.Conv7(x)
        return x


class Discriminator_x(nn.Module):
    def __init__(self):
        super(Discriminator_x, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(3, 32, 5, 1),
            nn.LeakyReLU(0.01),
        )
        self.Conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, 4, 2),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.Conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, 4, 1),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(128),
        )
        self.Conv4 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, 4, 2),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(256),
        )
        self.Conv5 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(256, 512, 4, 1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(512),
        )

    def forward(self, x):
        y = self.Conv1(x)
        y = self.Conv2(y)
        y = self.Conv3(y)
        y = self.Conv4(y)
        y = self.Conv5(y)
        return y


class Discriminator_z(nn.Module):
    def __init__(self):
        super(Discriminator_z, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(256, 512, 1, 1),
            nn.LeakyReLU(0.01),
        )
        self.Conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(512, 512, 1, 1),
            nn.LeakyReLU(0.01),
        )

    def forward(self, z):
        y = self.Conv1(z.view(-1, 256, 1, 1))
        y = self.Conv2(y)
        return y


class Discriminator_xz(nn.Module):
    def __init__(self):
        super(Discriminator_xz, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(1024, 1024, 1, 1),
            nn.LeakyReLU(0.01),
        )
        self.Conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(1024, 1024, 1, 1),
            nn.LeakyReLU(0.01),
        )
        self.Conv3 = nn.Sequential(
            nn.Dropout(0.2),  # I just can't put dropout after Sigmoid
            nn.Conv2d(1024, 1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, xz):
        y = self.Conv1(xz)
        y = self.Conv2(y)
        y = self.Conv3(y)
        return y


def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)  # isotropic gaussian
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
