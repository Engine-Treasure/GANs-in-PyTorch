# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from GAN.model3 import Generator, Discriminator
from utils import puzzle


# ==========
# settings
# ==========
img_size = 28
chkpt_dir = "checkpoints3"
sample_dir = "samples3"

# model parameters
batch_size = 16
epochs = 1000

g_lr = 1e-3
d_lr = 2e-4

nz = 74

# For convenience, use train set of MNIST in torchvision directly.
trainloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data/mnist', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    ),
    batch_size=batch_size, shuffle=True
)

G = Generator().cuda()
D = Discriminator().cuda()

BCE_loss = nn.BCELoss().cuda()

g_optimizer = optim.Adam(params=G.parameters(), lr=g_lr)
d_optimizer = optim.Adam(params=D.parameters(), lr=d_lr)

real_label = Variable(torch.ones(batch_size)).cuda()
fake_label = Variable(torch.zeros(batch_size)).cuda()

fixed_noise = Variable(torch.rand(36, nz)).cuda()

for epoch in range(epochs):
    for i, data in enumerate(trainloader, 0):

        # train Discriminator
        # ===================
        D.zero_grad()

        X = Variable(data[0]).cuda()
        z = Variable(torch.rand(batch_size, nz)).cuda()
        fakes = G.forward(z)

        d_output_real = D.forward(X)  # D(x)
        d_output_fake = D.forward(fakes)  # D(G(z))

        d_loss_real = BCE_loss(d_output_real, real_label)
        d_loss_fake = BCE_loss(d_output_fake, fake_label)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        d_x = d_output_real.data.mean()
        d_g_z1 = d_output_fake.data.mean()
        # ********************************

        # ===============
        # Train Generator
        # ===============
        G.zero_grad()

        z = Variable(torch.rand(batch_size, nz)).cuda()
        fakes = G.forward(z)

        g_output = D.forward(fakes)
        g_loss = BCE_loss(g_output, real_label)
        g_loss.backward()
        g_optimizer.step()

        d_g_z2 = g_output.data.mean()
        # ********************************

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, epochs, i, len(trainloader), d_loss.data[0], g_loss.data[0], d_x, d_g_z1, d_g_z2))

        if i % 1000 == 0:
            fake = G.forward(fixed_noise)
            fake_samples = fake.data.cpu().numpy().reshape(-1, img_size, img_size)
            img_arr = puzzle(fake_samples, n_raw=6)
            plt.imsave("{dir}/fake/fake_sample_{epoch}_{i}.png".format(dir=sample_dir, epoch=epoch, i=i),
                       img_arr, cmap="binary_r")
            # vutils.save_image(fake.data.view(-1, img_size, img_size),
            #                   "{dir}/fake/fake_sample_{epoch}_{i}.png".format(dir=sample_dir, epoch=epoch, i=i),
            #                   normalize=True)

    torch.save(G.state_dict(), "{dir}/Generator/g_epoch_{epoch}.chkpt".format(dir=chkpt_dir, epoch=epoch))
    torch.save(D.state_dict(), "{dir}/Discriminator/d_epoch_{epoch}.chkpt".format(dir=chkpt_dir, epoch=epoch))
