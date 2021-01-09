from sklearn.utils import shuffle
from copy import copy
from .helpers import assert_array_lens
from makiflow.core.debug import ExceptionScope


def cycle_generator(train_data, label_data, batch_size, shuffle_data=True):
    """
    Simple generator that cycles data infinitely. Each time the array of the data
    'has ended' the generator shuffles it.

    Parameters
    ----------
    train_data : list of arrays
        Contains training input data.
    label_data : list or arrays
        Contains training label data.
    batch_size : int
        The size of the batch.
    shuffle_data : bool
        Whether to shuffle the data.

    Returns
    -------
    python generator
    """
    assert isinstance(batch_size, int), 'Batch size must be an integer.'
    assert batch_size > 0, 'Batch size must be positive.'
    # Check whether all arrays have the same length
    with ExceptionScope('Train Data Check'):
        assert_array_lens(train_data)

    with ExceptionScope('Label Data Check'):
        assert_array_lens(label_data)

    if len(label_data) != 0:
        assert len(train_data[0]) == len(label_data[0]), ''

    n_batches = len(train_data[0]) // batch_size

    counter = 0
    while True:
        # Generate batches for input data
        train_batches = []
        for arr in train_data:
            batch = arr[counter * batch_size: (counter + 1) * batch_size]
            train_batches.append(batch)

        # Generate batches for label data
        label_batches = []
        for arr in label_data:
            batch = arr[counter * batch_size: (counter + 1) * batch_size]
            label_batches.append(batch)

        yield tuple(train_batches), tuple(label_batches)
        counter += 1

        if counter == n_batches:
            counter = 0
            if shuffle_data:
                n_train_arrs = len(train_data)
                data = shuffle(*(train_data + label_data))
                train_data, label_data = data[:n_train_arrs], data[n_train_arrs:]
