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

from __future__ import absolute_import
import tensorflow as tf
from makiflow.generators.pipeline.tfr.utils import _tensor_to_byte_feature

# Save form
SAVE_FORM = "{0}_{1}.tfrecord"


# Feature names
INPUT_X_FNAME = 'INPUT_X_FNAME'
TARGET_X_FNAME = 'TARGET_X_FNAME'
WEIGHT_MASK_FNAME = 'WEIGHT_MASK_FNAME'


# Serialize Object Detection Data Point
def serialize_regressor_data_point(input_tensor, target_tensor, weight_mask_tensor=None, sess=None):
    feature = {
        INPUT_X_FNAME: _tensor_to_byte_feature(input_tensor, sess),
        TARGET_X_FNAME: _tensor_to_byte_feature(target_tensor, sess)
    }

    if weight_mask_tensor is not None:
        feature[WEIGHT_MASK_FNAME] = _tensor_to_byte_feature(weight_mask_tensor, sess)

    features = tf.train.Features(feature=feature)
    example_proto = tf.train.Example(features=features)
    return example_proto.SerializeToString()


def record_regressor_train_data(input_tensors, target_tensors, weight_mask_tensors, tfrecord_path, sess=None):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for i, (input_tensor, target_tensor) in enumerate(zip(input_tensors, target_tensors)):

            if weight_mask_tensors is not None:
                weight_mask_tensor = weight_mask_tensors[i]
            else:
                weight_mask_tensor = None

            serialized_data_point = serialize_regressor_data_point(
                input_tensor=input_tensor,
                target_tensor=target_tensor,
                weight_mask_tensor=weight_mask_tensor,
                sess=sess
            )
            writer.write(serialized_data_point)


# Record data into multiple tfrecords
def record_mp_regressor_train_data(input_tensors, target_tensors, prefix,
                                   dp_per_record, weight_mask_tensors=None, sess=None):
    """
    Creates tfrecord dataset where each tfrecord contains `dp_per_second` data points

    Parameters
    ----------
    input_tensors : list or ndarray
        Array of input tensors.
    target_tensors : list or ndarray
        Array of target tensors.
    prefix : str
        Prefix for the tfrecords' names. All the filenames will have the same naming pattern:
        `prefix`_`tfrecord index`.tfrecord
    dp_per_record : int
        Data point per tfrecord. Defines how many tensors (locs, loc_masks, labels) will be
        put into one tfrecord file. It's better to use such `dp_per_record` that
        yields tfrecords of size 300-200 megabytes.
    sess : tf.Session
        In case if you can't or don't want to run TensorFlow eagerly, you can pass in the session object.
    weight_mask_tensors : list or ndarray
        Array of weight masks. By default equal to None, i. e. not used in recording data.
    """
    for i in range(len(input_tensors) // dp_per_record):
        input_tensor = input_tensors[dp_per_record * i: (i + 1) * dp_per_record]
        target_tensor = target_tensors[dp_per_record * i: (i + 1) * dp_per_record]

        if weight_mask_tensors is not None:
            weight_mask_tensor = weight_mask_tensors[dp_per_record * i: (i + 1) * dp_per_record]
        else:
            weight_mask_tensor = None

        tfrecord_name = SAVE_FORM.format(prefix, i)

        record_regressor_train_data(
            input_tensors=input_tensor,
            target_tensors=target_tensor,
            weight_mask_tensors=weight_mask_tensor,
            tfrecord_path=tfrecord_name,
            sess=sess
        )
