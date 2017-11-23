# -*- coding: utf-8 -*-

import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import fire

from GAN import model1, model2, model3
from utils import puzzle


class Sampler(object):
    def __init__(self, model, checkpoint):
        self.model= model

        if model == 1:
            self.G = model1.Generator()
        elif model == 2:
            self.G = model2.Generator()
        elif model == 3:
            self.G = model3.Generator()

        chkpt = torch.load("checkpoints{i}/Generator/g_epoch_{c}.chkpt".format(i=model, c=checkpoint))
        self.G.load_state_dict(chkpt)

    def sample(self, n):
        Z = Variable(torch.rand(n, 74)) if self.model == 3 else Variable(torch.rand(n, 100))

        samples = self.G.forward(Z)

        samples = samples.data.numpy().reshape(-1, 28, 28)
        img_grid = puzzle(samples, n_raw=3)

        plt.imshow(img_grid, cmap="binary_r")
        plt.show()

if __name__ == '__main__':
    fire.Fire(Sampler)
