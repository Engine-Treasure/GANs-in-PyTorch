# -*- coding: utf-8 -*-
import imageio

images = []
for epoch in range(30):
    filename = "samplesn/fake/{epoch}_0.png".format(epoch=epoch)
    images.append(imageio.imread(filename))
imageio.mimsave('samplesn/fake/fake.gif', images)
