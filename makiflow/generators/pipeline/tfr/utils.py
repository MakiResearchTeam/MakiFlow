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

import tensorflow as tf


def _bytes_feature(value):
    """
    TODO
    If you want to convert a tensor, you need to serialize it first.
    Use tf.io.serialize_tensor function.
    Parameters
    ----------
    value

    Returns
    -------

    """
    if tf.executing_eagerly():
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _tensor_to_byte_feature(tensor, sess=None):
    serialized_image = tf.io.serialize_tensor(tensor)
    if sess is not None:
        serialized_image = sess.run(serialized_image)
    elif sess is None and not tf.executing_eagerly():
        raise RuntimeError("TensorFlow is not executing eagerly. Please provide tf.Session to serialize tensors.")
    return _bytes_feature(serialized_image)


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

