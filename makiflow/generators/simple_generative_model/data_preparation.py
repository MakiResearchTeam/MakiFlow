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

# Feature names
INPUT_IMAGE_FNAME = 'INPUT_IMAGE_FNAME'
TARGET_IMAGE_FNAME = 'TARGET_IMAGE_FNAME'


# Serialize Object Detection Data Point
def serialize_sgm_data_point(input_image, target_image, sess=None):
    feature = {
        INPUT_IMAGE_FNAME: _tensor_to_byte_feature(input_image, sess),
        TARGET_IMAGE_FNAME: _tensor_to_byte_feature(target_image, sess)
    }
    features = tf.train.Features(feature=feature)
    example_proto = tf.train.Example(features=features)
    return example_proto.SerializeToString()


def record_sgm_train_data(input_images, target_images, tfrecord_path, sess=None):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for input_image, target_image in zip(input_images, target_images):
            serialized_data_point = serialize_sgm_data_point(input_image, target_image, sess)
            writer.write(serialized_data_point)


# Record data into multiple tfrecords
def record_mp_sgm_train_data(input_images, target_images, prefix, dp_per_record, sess=None):
    """
    Creates tfrecord dataset where each tfrecord contains `dp_per_second` data points.
    Parameters
    ----------
    input_images : list or ndarray
        Array of input images.
    target_images : list or ndarray
        Array of target images.
    prefix : str
        Prefix for the tfrecords' names. All the filenames will have the same naming pattern:
        `prefix`_`tfrecord index`.tfrecord
    dp_per_record : int
        Data point per tfrecord. Defines how many images (locs, loc_masks, labels) will be
        put into one tfrecord file. It's better to use such `dp_per_record` that
        yields tfrecords of size 300-200 megabytes.
    sess : tf.Session
        In case if you can't or don't want to run TensorFlow eagerly, you can pass in the session object.
    """
    for i in range(len(input_images) // dp_per_record):
        input_image = input_images[dp_per_record*i: (i+1)*dp_per_record]
        target_image = target_images[dp_per_record * i: (i + 1) * dp_per_record]
        tfrecord_name = f'{prefix}_{i}.tfrecord'
        record_sgm_train_data(
            input_images=input_image,
            target_images=target_image,
            tfrecord_path=tfrecord_name,
            sess=sess
        )
