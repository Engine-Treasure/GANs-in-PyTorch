# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable


class OneHotEncoder(nn.Module):
    def __init__(self, depth):
        super(OneHotEncoder, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth)

    def forward(self, X_in):
        X_in = X_in.long()
        return Variable(self.ones.index_select(0, X_in.data))

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


class Maxout(nn.Module):
    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m


if __name__ == '__main__':
    inputs = Variable(torch.LongTensor(range(10)))
    print(input)
    one_hot = OneHotEncoder(10)
    output = one_hot(input)
    print(output)

    input2 = Variable(torch.from_numpy(np.random.random((1, 784))))
    maxout = Maxout(784, 240, 5)
    output2 = maxout(input2.float())
    print(output2)

