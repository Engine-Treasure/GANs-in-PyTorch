# -*- coding: utf-8 -*-

from itertools import chain
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

# from ALI.model import Generator_x2z, Generator_z2x, Discriminator_x, Discriminator_z, Discriminator_xz
from ALI.models import create_models


class Generator_x2z(nn.Module):
    def __init__(self):
        super(Generator_x2z, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
        )
        self.Conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01),
        )
        self.Conv6 = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01),
        )
        self.Conv7 = nn.Sequential(
            nn.Conv2d(512, 256 * 2, 1, 1, bias=False),  # as the original code shows, 256 * 2
        )
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(512, 256)

    # reparameterization trick
    # https://github.com/pytorch/examples/blob/master/vae/main.py#L59
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            x = eps.mul(std).add_(mu)
            return x.view(x.size(0), -1, 1, 1)
        else:
            return mu.view(mu.size(0), -1, 1, 1)

    def forward(self, x):
        z = self.Conv1(x)
        z = self.Conv2(z)
        z = self.Conv3(z)
        z = self.Conv4(z)
        z = self.Conv5(z)
        z = self.Conv6(z)
        z = self.Conv7(z).view(-1, 512)
        mu, logvar = self.fc1(z), self.fc2(z)
        z = self.reparameterize(mu, logvar)
        return z


class Generator_z2x(nn.Module):
    def __init__(self):
        super(Generator_z2x, self).__init__()
        self.TConv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
        )
        self.TConv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
        )
        self.TConv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
        )
        self.TConv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
        )
        self.TConv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 5, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
        )
        self.Conv6 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, bias=False),

            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
        )
        self.Conv7 = nn.Sequential(
            nn.Conv2d(32, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.TConv1(z)
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
            nn.Dropout2d(0.2),
            nn.Conv2d(3, 32, 5, 1, bias=False),

            # https://github.com/edgarriba/ali-pytorch/blob/master/models.py#L120
            # nn.Conv2d(3, 32, 5, 1),
            nn.LeakyReLU(0.01),
        )
        self.Conv2 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 4, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
        )
        self.Conv3 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 4, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
        )
        self.Conv4 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 4, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
        )
        self.Conv5 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, 4, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.01),
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
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, 1, 1, bias=False),
            nn.LeakyReLU(0.01),
        )
        self.Conv2 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(512, 512, 1, 1, bias=False),
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
            nn.Dropout2d(0.2),
            nn.Conv2d(1024, 1024, 1, 1, bias=False),
            nn.LeakyReLU(0.01),
        )
        self.Conv2 = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(1024, 1024, 1, 1, bias=False),
            nn.LeakyReLU(0.01),
        )
        self.Conv3 = nn.Sequential(
            nn.Dropout2d(0.2),  # I just can't put dropout after Sigmoid
            nn.Conv2d(1024, 1, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, xz):
        y = self.Conv1(xz)
        y = self.Conv2(y)
        y = self.Conv3(y)
        return y.view(-1, 1)


def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.01)  # isotropic gaussian
            # m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.01)
            m.bias.data.zero_()


def reset_grad():
    D_xz.zero_grad()
    D_x.zero_grad()
    D_z.zero_grad()
    G_x2z.zero_grad()
    G_z2x.zero_grad()


def log(x_):
    return torch.log(x_ + 1e-8)


def softplus(x_):
    return torch.log(1.0 + torch.exp(x_))


img_size = 32
chkpt_dir = "checkpoints"
sample_dir = "samples"
reconstruction_dir = "reconstructions"
real_dir = "reals"

batch_size = 100  # it is 100 in the paper
epochs = 100
eps = 1e-8

G_x2z = Generator_x2z().cuda()  # also the encoder, the Inference Network
G_z2x = Generator_z2x().cuda()  # also the decoder, the Generative Network
# D_x and D_z are separated from D_xz, only to make the architecture of Discriminator more clear.
D_x = Discriminator_x().cuda()
D_z = Discriminator_z().cuda()
D_xz = Discriminator_xz().cuda()

# init weights and biases
init_weights(G_x2z)
init_weights(G_z2x)
init_weights(D_x)
init_weights(D_z)
init_weights(D_xz)

lr = 1e-4
betas = (0.5, 1e-3)
g_params = chain(G_x2z.parameters(), G_z2x.parameters())
d_params = chain(D_x.parameters(), D_z.parameters(), D_xz.parameters())
g_optimizer = optim.Adam(g_params, lr=lr, betas=betas)
d_optimizer = optim.Adam(d_params, lr=lr, betas=betas)

bce = nn.BCELoss().cuda()

trainloader = torch.utils.data.DataLoader(
    datasets.SVHN(
        '../data/svhn', split="train", download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    ),
    batch_size=batch_size, shuffle=True
)

testloader = torch.utils.data.DataLoader(
    datasets.SVHN(
        '../data/svhn', split="test", download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    ),
    batch_size=batch_size, shuffle=True
)

fixed_noise = Variable(torch.randn(10, 256), volatile=True).cuda()


# chkpt_gx = torch.load(os.path.join("checkpoints", "G_x2z", "64.pkl"))
# G_x2z.load_state_dict(chkpt_gx)
# chkpt_gz = torch.load(os.path.join("checkpoints", "G_z2x", "64.pkl"))
# G_z2x.load_state_dict(chkpt_gz)
# chkpt_dx = torch.load(os.path.join("checkpoints", "D_x", "64.pkl"))
# D_x.load_state_dict(chkpt_dx)
# chkpt_dz = torch.load(os.path.join("checkpoints", "D_z", "64.pkl"))
# D_z.load_state_dict(chkpt_dz)
# chkpt_dxz = torch.load(os.path.join("checkpoints", "D_xz", "64.pkl"))
# D_xz.load_state_dict(chkpt_dxz)

def train(trainloader, epoch):
    D_xz.train(), D_x.train(), D_z.train()
    G_x2z.train(), G_z2x.train()

    for batch_idx, (x, _) in enumerate(trainloader):
        # if batch_idx == 10:
        #     return
        batch_size = x.size(0)

        x = Variable(x).cuda()
        z = Variable(torch.randn(batch_size, 256, 1, 1)).cuda()
        # epsilon = Variable(torch.randn(batch_size, 1, 1, 1)).cuda()

        x_tilde = G_z2x(z)  # p(x|z)

        # q(z|x) = N(mu, sigma^2 I), reparameterize
        # see https://github.com/IshmaelBelghazi/ALI/blob/master/ali/conditional_bricks.py#L154
        z_hat = G_x2z(x)  # q(z|x)

        # mu, sigma = z_hat[:, :256], z_hat[:, 256:].exp()
        # z_hat = mu + sigma * epsilon[:batch_size].expand_as(sigma)

        xx = D_x(x)
        zhat_zhat = D_z(z_hat)
        xtilde_xtilde = D_x(x_tilde)
        zz = D_z(z)

        # print("ZHAT", zhat_zhat)

        d_data_preds = D_xz(torch.cat([xx, zhat_zhat], dim=1)) + eps  # real, encoder,
        d_sample_preds = D_xz(torch.cat([xtilde_xtilde, zz], dim=1)) + eps  # fake, decoder

        real_label = Variable(torch.ones(batch_size) - 0.1).cuda()
        fake_label = Variable(torch.zeros(batch_size)).cuda()
        # d_loss_real = bce(d_data_preds, real_label)
        # d_loss_fake = bce(d_sample_preds, fake_label)
        d_x_zhat = d_data_preds.data.mean()
        d_xtilde_z = d_sample_preds.data.mean()
        # d_loss = d_loss_real + d_loss_fake

        d_loss = -torch.mean(log(d_data_preds) + log(1 - d_sample_preds))
        # d_loss = torch.mean(softplus(-d_data_preds) + softplus(d_sample_preds))

        reset_grad()
        d_loss.backward(retain_graph=True)  # the gradients will flow all networks: D_xz, D_x, D_z, G_x2z, G_z2x
        d_optimizer.step()

        reset_grad()
        g_loss = -torch.mean(log(d_sample_preds) + log(1 - d_data_preds))
        g_loss.backward()
        g_optimizer.step()
        # ********************************

        # # ===============
        # # Train Generator
        # # ===============
        # z = Variable(torch.randn(batch_size, 256, 1, 1)).cuda()
        # x_tilde = G_z2x(z)
        #
        # # to avoid the RuntimeError `backward through the graph a second time`,
        # # compute z_hat again, though the result remains unchanged
        # z_hat = G_x2z(x)
        # # mu, sigma = z_hat[:, :256], z_hat[:, 256:].exp()
        # # z_hat = mu + sigma * epsilon[:batch_size].expand_as(sigma)
        #
        # g_data_preds = D_xz(torch.cat([D_x(x), D_z(z_hat)], dim=1)) + eps  # real
        # g_sample_preds = D_xz(torch.cat([D_x(x_tilde), D_z(z)], dim=1)) + eps  # fake
        #
        # g_x_zhat = g_data_preds.data.mean()
        # g_xtilde_z = g_sample_preds.data.mean()
        #
        # # Unlike the previous training process (see GAN and cGAN),
        # # we need to update 2 networks, G_x2z and Gz2x
        # # g_loss_real = bce(g_sample_preds.clamp(min=0.01, max=0.99), real_label)  # when train Gs, exchange the labels
        # # g_loss_fake = bce(g_data_preds.clamp(min=0.01, max=0.99), fake_label)
        # # g_loss_real = bce(g_sample_preds, real_label)  # when train Gs, exchange the labels
        # # g_loss_fake = bce(g_data_preds, fake_label)
        # # g_loss = g_loss_fake + g_loss_real
        #
        # g_loss = -torch.mean(log(g_sample_preds) + log(1 - g_data_preds))
        # # g_loss = torch.mean(softplus(g_data_preds) + softplus(-g_sample_preds))
        #
        # reset_grad()
        # g_loss.backward()
        # g_optimizer.step()
        # # ********************************


        print(
            "[{e}/{epochs}] [{i}/{steps}] Loss D: {loss_d:.6f}; Loss G: {loss_g:.6f};D(x, zhat): {d1:.6f}; D(xhat, z): {d2:.6f}".format(
                e=e, epochs=epochs, i=batch_idx,
                steps=len(trainloader),
                loss_d=d_loss.data[0],
                loss_g=g_loss.data[0],
                d1=d_x_zhat, d2=d_xtilde_z))
                # g1=g_x_zhat, g2=g_xtilde_z


def dev(testloader, epoch):
    G_x2z.eval(), G_z2x.eval()
    x, _ = iter(testloader).next()

    z_hat = G_x2z(Variable(x, volatile=True).cuda())
    # mu, sigma = z_hat[:, :256], z_hat[:, 256:].exp()
    # z_hat = mu + sigma * epsilon.expand_as(sigma)
    recon = G_z2x(z_hat)

    # mu, sigma = z_hat.cpu()[:, :256], z_hat.cpu()[:, 256:].exp()
    # recon = G_z2x(z_hat)

    save_image(torch.cat([x[:10], recon.cpu().data[:10]], dim=0), "samples/{e}.png".format(e=epoch), nrow=10)


for e in range(epochs):
    train(trainloader, e)
    dev(testloader, e)

    torch.save(G_x2z.state_dict(), "{dir}/G_x2z/{epoch}.pkl".format(dir=chkpt_dir, epoch=e))
    torch.save(G_z2x.state_dict(), "{dir}/G_z2x/{epoch}.pkl".format(dir=chkpt_dir, epoch=e))
    torch.save(D_x.state_dict(), "{dir}/D_x/{epoch}.pkl".format(dir=chkpt_dir, epoch=e))
    torch.save(D_z.state_dict(), "{dir}/D_z/{epoch}.pkl".format(dir=chkpt_dir, epoch=e))
    torch.save(D_xz.state_dict(), "{dir}/D_xz/{epoch}.pkl".format(dir=chkpt_dir, epoch=e))
