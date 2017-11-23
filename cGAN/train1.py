# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from cGAN.model1 import Generator, Discriminator
from utils import puzzle

# ==========
# settings
# ==========
img_size = 28
chkpt_dir = "checkpoints"
sample_dir = "samples"

# model parameters
batch_size = 100
epochs = 1000

momentum = 0.7
lr = 0.1
decay_factor = 1.00004
lr_min = 0.000001

nz = 100

# For convenience, use train set of MNIST in torchvision directly.
trainloader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=True)

# TODO: best estimate of log-likelihood on the validation set was used as stopping point
validloader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=True
)

G = Generator()
D = Discriminator()

criterion = nn.BCELoss()
g_optimizer = optim.SGD(params=G.parameters(), lr=lr, momentum=momentum)
d_optimizer = optim.SGD(params=D.parameters(), lr=lr, momentum=momentum)

g_scheduler = optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=1 / decay_factor)
d_scheduler = optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=1 / decay_factor)

real_label = Variable(torch.ones(batch_size))
fake_label = Variable(torch.zeros(batch_size))

fixed_z = Variable(torch.rand(100, nz))
fixed_condition = Variable(torch.from_numpy(np.array([i // 10 for i in range(100)])))

# The training process refers to DCGANs in PyTorch Example
for epoch in range(epochs):
    for i, data in enumerate(trainloader, 0):
        g_scheduler.step()  # LR Scheduler only update learning rate!!!
        d_scheduler.step()

        # ================================
        # train Discriminator
        D.zero_grad()

        # data[0] - image, data[1] - target, e.t. condition
        X, y = Variable(data[0].view(batch_size, img_size * img_size)), Variable(data[1])
        z = Variable(torch.rand(batch_size, nz))
        fakes = G.forward(z, y)

        d_output_real = D.forward(X, y)
        d_output_fake = D.forward(fakes, y)

        d_loss_real = criterion(d_output_real, real_label)
        d_loss_fake = criterion(d_output_fake, fake_label)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        d_x = d_output_real.data.mean()  # mean of predictions on true samples
        d_g_z1 = d_output_fake.data.mean()
        # ********************************

        # ================================
        # Train Generator
        G.zero_grad()

        z = Variable(torch.rand(batch_size, nz))
        fakes = G.forward(z, y)

        g_output = D.forward(fakes, y)
        g_loss = criterion(g_output, real_label)
        g_loss.backward()
        g_optimizer.step()

        d_g_z2 = g_output.data.mean()
        # ********************************

        print('[%d/%d][%d/%d] LR: %f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, epochs, i, len(trainloader), d_optimizer.param_groups[0]["lr"],
                 d_loss.data[0], g_loss.data[0], d_x, d_g_z1, d_g_z2))

        if i % 100 == 0:
            fake = G.forward(fixed_z, fixed_condition)
            fake_samples = fake.data.cpu().numpy().reshape(-1, img_size, img_size)
            img_arr = puzzle(fake_samples, n_raw=10)
            plt.imsave("{dir}/fake/fake_sample_{epoch}_{i}.png".format(dir=sample_dir, epoch=epoch, i=i),
                       img_arr, cmap="binary_r")

    torch.save(G.state_dict(), "{dir}/Generator/g_epoch_{epoch}.chkpt".format(dir=chkpt_dir, epoch=epoch))
    torch.save(D.state_dict(), "{dir}/Discriminator/d_epoch_{epoch}.chkpt".format(dir=chkpt_dir, epoch=epoch))
