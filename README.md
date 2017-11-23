GANs in PyTorch
---

Here are some popular variants of GANs, implemented in PyTorch. Moreover, I try to implement them following their paper as much as possible. And you could see what they look like and how they work when they are first proposed.

* [GAN](GAN) - The original one.
    * [The very origianl one](GAN/model1.py)
    * [The original one with additional BN layers in generator](GAN/model2.py)
    * [GAN that described in the InfoGAN paper](GAN/model3.py)
* [cGAN](cGAN) - Conditional GAN.
    * [Conditional GAN without BN layers in generator](cGAN/model1.py)
    * [Conditional GAN with BN layers in generator](cGAN/model2.py)
    