# -*- coding: utf-8 -*-
import numpy as np

import torch.nn as nn


def puzzle(samples, n_raw=3):
    """
    :param samples: seq of samples
    :param n_raw: number of raws of the result image
    """
    lines = []  # line of samples
    for c in range(0, samples.shape[0]):
        if c % n_raw == 0:
            lines.append([])
        lines[c // n_raw].append(samples[c])

    grid = []  # array of samples
    for r in lines:
        grid.append(np.hstack(r))

    return np.vstack(grid)


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
