import imageio

images = []
for epoch in range(50):
        filename = "samples2/fake/fake_sample_{epoch}_500.png".format(epoch=epoch)
        images.append(imageio.imread(filename))
imageio.mimsave('samples2/fake/fake.gif', images)