# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from GAN.model1 import Generator, Discriminator
from utils import puzzle


def g_weights_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-0.05, 0.05)
        # m.bias.data.fill_(0)
    elif isinstance(m, nn.Sigmoid):
        # TODO: init_sigmoid_bias_from_marginals
        pass


def d_weights_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-0.005, 0.005)
        # m.bias.data.fill_(0)


# ==========
# settings
# ==========
img_size = 28
chkpt_dir = "checkpoints"
sample_dir = "samples"

# model parameters
batch_size = 100
epochs = 10000

momentum = 0.5
momentum_increase = (0.7 - 0.5) / 250
momentum_max = 0.7

lr = 0.1
decay_factor = 1.000004
lr_min = 0.000001

nz = 100

# For convenience, use train set of MNIST in torchvision directly.
trainloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data/mnist', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    ),
    batch_size=batch_size, shuffle=False
)

G = Generator().cuda()
D = Discriminator().cuda()
G.apply(g_weights_init)  # init network weights
D.apply(d_weights_init)

BCE_loss = nn.BCELoss().cuda()

g_optimizer = optim.SGD(params=G.parameters(), lr=lr, momentum=momentum)
d_optimizer = optim.SGD(params=D.parameters(), lr=lr, momentum=momentum)

g_scheduler = optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=1 / decay_factor)
d_scheduler = optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=1 / decay_factor)

real_label = Variable(torch.ones(batch_size)).cuda()
fake_label = Variable(torch.zeros(batch_size)).cuda()

fixed_noise = Variable(torch.rand(9, nz), volatile=True).cuda()

for epoch in range(epochs):
    for i, data in enumerate(trainloader, 0):
        if i == 500:  # train with 0~50000 samples, consistent with original setting
            break

        # as Pylearn2's comment says, exponential decay is callback for sgd algorithm and is executed mini-batch,
        # so we keep scheduler stepping here
        g_scheduler.step()
        d_scheduler.step()

        # ===================
        # train Discriminator
        # ===================
        D.zero_grad()

        X = Variable(data[0].view(batch_size, img_size * img_size)).cuda()
        z = Variable(torch.rand(batch_size, nz)).cuda()
        fakes = G.forward(z)

        d_output_real = D.forward(X)  # D(x)
        d_output_fake = D.forward(fakes)  # D(G(z))

        d_loss_real = BCE_loss(d_output_real, real_label)
        d_loss_fake = BCE_loss(d_output_fake, fake_label)
        d_loss = d_loss_real + d_loss_fake
        # d_loss = -torch.mean(torch.log(d_output_real) + torch.log(1 - d_output_fake))
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

        print('[%d/%d][%d/%d] LR: %f Momentum: %f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, epochs, i, len(trainloader), d_optimizer.param_groups[0]["lr"],
                 d_optimizer.param_groups[0]["momentum"],
                 d_loss.data[0], g_loss.data[0], d_x, d_g_z1, d_g_z2))

        if i % 100 == 0:
            fake = G.forward(fixed_noise)
            fake_samples = fake.data.cpu().numpy().reshape(-1, img_size, img_size)
            img_arr = puzzle(fake_samples)
            plt.imsave("{dir}/fake/fake_sample_{epoch}_{i}.png".format(dir=sample_dir, epoch=epoch, i=i),
                       img_arr, cmap="binary_r")

    # as Pylearn2's comment says `Updates the momentum on algorithm based on the epochs elapsed.`
    # so we keep momentum adjusting here outside the inner loop.
    if g_optimizer.param_groups[0]["momentum"] < momentum_max:
        g_optimizer.param_groups[0]["momentum"] += momentum_increase
        d_optimizer.param_groups[0]["momentum"] += momentum_increase

    torch.save(G.state_dict(), "{dir}/Generator/g_epoch_{epoch}.chkpt".format(dir=chkpt_dir, epoch=epoch))
    torch.save(D.state_dict(), "{dir}/Discriminator/d_epoch_{epoch}.chkpt".format(dir=chkpt_dir, epoch=epoch))
