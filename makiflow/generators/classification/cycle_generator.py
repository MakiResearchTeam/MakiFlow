from sklearn.utils import shuffle
from copy import copy


def cycle_generator(Xtrain, Ytrain, batch_size, shuffle_data=True):
    """
    Simple generator that cycles data infinitely. Each time the array of the data
    'has ended' the generator shuffles it.
    Parameters
    ----------
    Xtrain : list or ndarray
        Array of training images.
    Ytrain : list or ndarray
        Array of training labels
    batch_size : int
        The size of the batch.
    shuffle : bool
        Whether to shuffle the data.

    Returns
    -------
    python generator
    """
    assert len(Xtrain) == len(Ytrain)
    n_batches = len(Xtrain) // batch_size

    Xtrain = copy(Xtrain)
    Ytrain = copy(Ytrain)

    counter = 0
    while True:
        Xbatch = Xtrain[counter * batch_size: (counter + 1) * batch_size]
        Ybatch = Ytrain[counter * batch_size: (counter + 1) * batch_size]

        yield (Xbatch,), (Ybatch,)
        counter += 1

        if counter == n_batches:
            counter = 0
            if shuffle_data:
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
