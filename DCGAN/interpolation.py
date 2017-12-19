# -*- coding: utf-8 -*-

import os

import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from DCGAN.model import Generator

z1 = torch.randn(1, 100, 1, 1)
z2 = torch.randn(1, 100, 1, 1)
step = (z2 - z1) / 36
zn = torch.cat([z1 + step * i for i in range(36)], dim=0)

G = Generator()
g_checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "checkpointsn", "G", "30.pth"))
G.load_state_dict(g_checkpoint)


samples = G.forward(Variable(zn))
save_image(samples.data, 'interpolation36.png', normalize=True, nrow=9)
