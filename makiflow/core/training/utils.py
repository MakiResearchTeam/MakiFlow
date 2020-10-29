# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
from makiflow.core import MakiTensor

EPOCH = 'Epoch:'


def print_train_info(epoch, *args):
    output = ''
    for value_name, value in args:
        value_name = value_name.lower() + ': '
        value_name = value_name.title()
        output += value_name + '{:0.4f}'.format(value) + ' '

    print(EPOCH, epoch, output)


def moving_average(old_val, new_val, iteration):
    if iteration == 0:
        return new_val
    else:
        return old_val * 0.9 + new_val * 0.1


def new_optimizer_used():
    print('New optimizer is used.')


def loss_is_built():
    print('Loss is built.')


def pack_data(feed_dict_config, data: list):
    """
    Packs data into a dictionary with pairs (tf.Tensor, data).
    This dictionary is then used as the `feed_dict` argument in the session.run() method.
    Parameters
    ----------
    feed_dict_config : dict
        Contains pairs (MakiTensor, int) or (tf.Tensor, int), where int is the index of the data point in the `data`.
    data : list
        The data to pack.

    Returns
    -------
    dict
        Dictionary with packed data.
    """

    feed_dict = dict()
    for t, i in feed_dict_config.items():
        # If the `t` is a tf.Tensor
        data_tensor = t
        if isinstance(t, MakiTensor):
            data_tensor = t.get_data_tensor()
        feed_dict[data_tensor] = data[i]
    return feed_dict


class IteratorCloser:
    def __init__(self, raise_exception=False):
        """
        Used for tqdm loops. If a tqdm loop was closed by unnatural means (exception),
        in most case its print breaks. To avoid such situation, one must always close
        the tqdm iterator.
        Just wrap the cycle code using `with` statement.
        Parameters
        ----------
        raise_exception : bool
            Set to True if you want an exception raised in a loop to be re-raised after `with`.
        """
        self._raise_exception = raise_exception
        self._tqdm_iterator = None

    def __enter__(self):
        pass

    def set_iterator(self, it):
        self._tqdm_iterator = it

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._tqdm_iterator is not None:
            self._tqdm_iterator.close()

        if exc_type is not None:
            print(f'Exception type: {exc_type}')
            print(f'Exception values: {exc_val}')

        # An exception is not raised if True is returned
        return not self._raise_exception
