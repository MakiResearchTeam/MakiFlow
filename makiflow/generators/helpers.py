import numpy as np


def data_iterator(*args, batch_size=1):
    """
    Iterates over the array yielding batches of the given size.

    Parameters
    ----------
    *args : arraylike
        An array to iterate over.
    batch_size : int
        The batch size.

    Returns
    -------
    iterator
    """
    assert isinstance(batch_size, int), 'Batch size must be an integer.'
    assert batch_size > 0, 'Batch size must be positive.'
    # Check whether all arrays have the same length
    for i, arr1 in enumerate(args):
        for j, arr2 in enumerate(args):
            assert len(arr1) == len(arr2), f'All arrays must have the same length, but array {i} and array {j} ' \
                f'have lengths of {len(arr1)} and {len(arr2)} respectively.'

    iterations = len(args[0]) // batch_size
    for i in range(iterations):
        # Take batches from each of the arrays
        batches = []
        for arr in args:
            batches.append(arr[i*batch_size: (i+1)*batch_size])

        yield tuple(batches)

    if iterations * batch_size != len(args[0]):
        i = iterations
        to_add = batch_size - len(args[0]) % batch_size
        print('toadd', to_add)
        batches = []
        for arr in args:
            batch = arr[i * batch_size:]
            print(len(batch))

            if isinstance(batch, list):
                updated = batch + to_add * batch[-1:]
            elif isinstance(batch, np.ndarray):
                updated = np.concatenate([batch] + to_add * [batch[-1:]], axis=0)
            else:
                raise ValueError(f'Unknown data structure. Expected ndarray or list, received {type(batch)}')

            batches.append(updated)

        yield tuple(batches)


if __name__ == '__main__':
    from makiflow.core.debug import DebugContext
    a, b = np.ones(4), np.ones(4) * 2
    for one, two in data_iterator(a, b, batch_size=2):
        print(one, two)

    with DebugContext('Not equal arrays.'):
        a, b = np.ones(4), np.ones(3) * 2
        for one, two in data_iterator(a, b, batch_size=2):
            print(one, two)

    with DebugContext('Array length and batch size are not multiples.'):
        a, b = np.ones(4), np.ones(4) * 2
        for one, two in data_iterator(a, b, batch_size=3):
            print(one, two)
            assert len(one) == 3 and len(two) == 3

    with DebugContext('Lists usage.'):
        a, b = np.ones(4), np.ones(4) * 2
        for one, two in data_iterator(a.tolist(), b.tolist(), batch_size=3):
            print(one, two)
