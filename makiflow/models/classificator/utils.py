import matplotlib.pyplot as plt
import numpy as np


plt.switch_backend('agg')
# For loading dendrite images


def error_rate(Y, T):
    return np.mean(Y != T)


def cross_entropy(Y, T):
    return -np.mean(T * np.log(Y))


def sparse_cross_entropy(Y, T):
    return -np.mean(np.log(Y[np.arange(Y.shape[0]), T]))
