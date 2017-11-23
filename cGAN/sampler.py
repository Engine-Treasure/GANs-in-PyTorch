# -*- coding: utf-8 -*-

import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import fire

from cGAN import model1, model2


class Sampler(object):
    def __init__(self, model, checkpoint):
        self.model = model

        if model == 1:
            self.G = model1.Generator()
        elif model == 2:
            self.G = model2.Generator()

        chkpt = torch.load("checkpoints{i}/Generator/g_epoch_{c}.chkpt".format(i=model, c=checkpoint))
        self.G.load_state_dict(chkpt)

    def sample(self, number):
        Z = Variable(torch.rand(1, 100))
        Y = Variable(torch.from_numpy(np.array([number])))

        sample = self.G.forward(Z, Y).view(28, 28)

        plt.imshow(sample.data.numpy(), cmap="binary_r")
        plt.show()


if __name__ == '__main__':
    fire.Fire(Sampler)
