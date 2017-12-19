# -*- coding: utf-8 -*-
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from DCGAN.model import Discriminator, Generator, init_weights

batch_size = 32
nz = 100
img_size = 64
samples = "samplesn"
checkpoints = "checkpointsn"

dataset = datasets.LSUN(db_path="../data/lsun", classes=['church_outdoor_train'],
                        transform=transforms.Compose([
                            transforms.Scale(img_size),
                            transforms.CenterCrop(img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

G = Generator().cuda()
G.apply(init_weights)
D = Discriminator().cuda()
D.apply(init_weights)


# g_checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "checkpoints", "G", "48.pth"))
# G.load_state_dict(g_checkpoint)
# d_checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "checkpoints", "D", "48.pth"))
# D.load_state_dict(d_checkpoint)

criterion = nn.BCELoss().cuda()

# setup optimizer
d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = Variable(torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1).cuda())

for epoch in range(100):
    for i, (x, _) in enumerate(dataloader, 0):
        # Alternating Gradient Descent
        b_size = x.size(0)

        # ===================
        # train Discriminator
        # ===================
        D.zero_grad()
        x = x.cuda()
        # The paper uses unifrom distribution, maybe it is better to use gaussian distribution
        z = torch.randn(b_size, nz, 1, 1).cuda()
        # z = torch.rand(b_size, nz, 1, 1).cuda()
        fake = G(Variable(z))
        real_labels = Variable(torch.ones(b_size).cuda() - 0.1)  # one-sided smoonth
        fake_labels = Variable(torch.zeros(b_size).cuda())

        # alternative loss
        d_real = D(Variable(x))
        d_loss_real = criterion(d_real, real_labels)
        d_loss_real.backward()

        d_fake = D(fake.detach())
        d_loss_fake = criterion(d_fake, fake_labels)
        d_loss_fake.backward()

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.step()

        d_x = d_real.data.mean()
        d_g_z1 = d_fake.data.mean()
        # ********************************************************

        # ===================
        # train Generator
        # ===================
        G.zero_grad()

        g_output = D(fake)  # train with updated Discriminator
        g_loss = criterion(g_output, real_labels)  # G tries to fool D
        g_loss.backward()

        d_g_z2 = g_output.data.mean()
        g_optimizer.step()
        # ********************************************************

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, 100, i, len(dataloader),
                 d_loss.data[0], g_loss.data[0], d_x, d_g_z1, d_g_z2))
        if i % 100 == 0:
            save_image(x, '{dir}/real/{e}_{i}.png'.format(dir=samples, e=epoch, i=i), normalize=True)
            fake = G(fixed_noise)
            save_image(fake.data, '{dir}/fake/{e}_{i}.png'.format(dir=samples, e=epoch, i=i), normalize=True)

    # do checkpointing
    torch.save(G.state_dict(), '{dir}/G/{e}.pth'.format(dir=checkpoints, e=epoch))
    torch.save(D.state_dict(), '{dir}/D/{e}.pth'.format(dir=checkpoints, e=epoch))
