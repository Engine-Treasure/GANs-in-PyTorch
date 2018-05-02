# -*- coding: utf-8 -*-

"""
Copied from Sebastian Raschka's Notebook:
https://nbviewer.jupyter.org/github/rasbt/pattern_classification/blob/master/parameter_estimation_techniques/parzen_window_technique.ipynb#Size-of-the-training-data-set
"""

import numpy as np
from scipy.stats import gaussian_kde


def hypercube_kernel(h, x, x_i, d):
    """
    Implementation of a hypercube kernel for Parzen-window estimation.

    Keyword arguments:
        h: window width
        x: point x for density estimation, 'd x 1'-dimensional numpy array
        x_i: point from training sample, 'd x 1'-dimensional numpy array

    Returns a 'd x 1'-dimensional numpy array as input for a window function.

    """
    assert (x.shape == x_i.shape), 'vectors x and x_i must have the same dimensions'
    return (x - x_i) / (h)


def gaussian_kernel(h, x, x_i, d):
    """
    Implementation of a hypercube kernel for Parzen-window estimation.

    Keyword arguments:
        h: window width
        x: point x for density estimation, 'd x 1'-dimensional numpy array
        x_i: point from training sample, 'd x 1'-dimensional numpy array

    Returns a 'd x 1'-dimensional numpy array as input for a window function.

    """
    assert (x.shape == x_i.shape), 'vectors x and x_i must have the same dimensions'
    return np.exp(-((x - x_i) / h) ** 2 / 2) / ((np.sqrt(2) * np.pi * h) ** d)


def parzen_window_func(x_vec, h=1):
    """
    Implementation of the window function. Returns 1 if 'd x 1'-sample vector
    lies within inside the window, 0 otherwise.

    """
    for row in x_vec:
        if np.abs(row) > (1 / 2):
            return 0
    return 1


def parzen_estimation(x_samples, point_x, h, d, window_func, kernel_func):
    """
    Implementation of a parzen-window estimation.

    Keyword arguments:
        x_samples: A 'n x d'-dimensional numpy array, where each sample
            is stored in a separate row. (= training sample)
        point_x: point x for density estimation, 'd x 1'-dimensional numpy array
        h: window width
        d: dimensions
        window_func: a Parzen window function (phi)
        kernel_function: A hypercube or Gaussian kernel functions

    Returns the density estimate p(x).

    """
    k_n = 0
    for row in x_samples:
        x_i = kernel_func(h=h, x=point_x, x_i=row[:, np.newaxis], d=d)
        k_n += window_func(x_i, h=h)
    return (k_n / len(x_samples)) / (h ** d)

if __name__ == '__main__':
    import time

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    mu_vec = np.array([0, 0])
    cov_mat = np.eye(2)
    x_2dgauss = np.random.multivariate_normal(mu_vec, cov_mat, 10000)

    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)

    gkde = gaussian_kde(x_2dgauss)
    start = time.time()
    density = gkde.evaluate(np.array([X.ravel(), Y.ravel()]))
    print(time.time() - start)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca(projection='3d')

    Z = []

    start = time.time()
    for i, j in zip(X.ravel(), Y.ravel()):
        s = time.time()
        Z.append(parzen_estimation(x_2dgauss, np.array([[i], [j]]), h=0.3, d=2,
                                   window_func=parzen_window_func,
                                   kernel_func=gaussian_kernel))
        print(time.time() - s)

    print(time.time() - start)

    Z = np.asarray(Z).reshape(100, 100)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_zlim(0, 0.2)

    ax.zaxis.set_major_locator(plt.LinearLocator(10))
    ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('p(x)')

    plt.title('Hypercube kernel with window width h=0.3')

    fig.colorbar(surf, shrink=0.5, aspect=7, cmap=plt.cm.coolwarm)

    plt.show()

