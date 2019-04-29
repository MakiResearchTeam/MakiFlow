import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

plt.switch_backend('agg')
# For loading dendrite images


def error_rate(Y, T):
    return np.mean(Y != T)


def get_data():
    df = pd.read_csv('/home/student401/study/data_sets/mnist/train.csv')
    data = df.values

    X = data[:, 1:]
    Y = data[:, 0]

    return X, Y


def one_hot_encoding(Y):
    N = len(Y)
    D = len(set(Y))
    new_Y = np.zeros((N, D))

    for i in range(N):
        new_Y[i, Y[i]] = 1

    return new_Y


# For denrites
def one_hot_encoding_d(Y):
    N = len(Y)
    D = len(set(Y))
    new_Y = np.zeros((N, D))

    for i in range(N):
        new_Y[i, Y[i] - 1] = 1

    return new_Y


def get_preprocessed_data():
    X, Y = get_data()
    X = X / 255
    Y = one_hot_encoding(Y)
    X, Y = shuffle(X, Y)
    return X, Y


def rearrange_tf(X):
    # input is (N, 784)
    # output is (N, 28, 28, 1)
    new_X = np.zeros((X.shape[0], 28, 28, 1))
    for j in range(28):
        new_X[:, j, :, 0] += X[:, j * 28:(j + 1) * 28]

    return new_X


def get_preprocessed_image_data_tf():
    X, Y = get_preprocessed_data()
    X = rearrange_tf(X)
    return X, Y


def cross_entropy(Y, T):
    return -np.mean(T * np.log(Y))


def sparse_cross_entropy(Y, T):
    return -np.mean(np.log(Y[np.arange(Y.shape[0]), T]))
