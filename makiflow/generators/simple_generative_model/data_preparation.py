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
INPUT_IMAGE_FNAME = 'INPUT_IMAGE_FNAME'
TARGET_IMAGE_FNAME = 'TARGET_IMAGE_FNAME'
WEIGHT_MASK_FNAME = 'WEIGHT_MASK_FNAME'


# Serialize Object Detection Data Point
def serialize_sgm_data_point(input_image, target_image, weight_mask_image=None, sess=None):
    feature = {
        INPUT_IMAGE_FNAME: _tensor_to_byte_feature(input_image, sess),
        TARGET_IMAGE_FNAME: _tensor_to_byte_feature(target_image, sess)
    }

    if weight_mask_image is not None:
        feature[WEIGHT_MASK_FNAME] = _tensor_to_byte_feature(weight_mask_image, sess)

    features = tf.train.Features(feature=feature)
    example_proto = tf.train.Example(features=features)
    return example_proto.SerializeToString()


def record_sgm_train_data(input_images, target_images, weight_mask_images, tfrecord_path, sess=None):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for i, (input_image, target_image) in enumerate(zip(input_images, target_images)):

            if weight_mask_images is not None:
                weight_mask_image = weight_mask_images[i]
            else:
                weight_mask_image = None

            serialized_data_point = serialize_sgm_data_point(input_image=input_image,
                                                             target_image=target_image,
                                                             weight_mask_image=weight_mask_image,
                                                             sess=sess
            )
            writer.write(serialized_data_point)


# Record data into multiple tfrecords
def record_mp_sgm_train_data(input_images, target_images, prefix, dp_per_record, weight_mask_images=None, sess=None):
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
    weight_mask_images : list or ndarray
        Array of weight masks. By default equal to None, i. e. not used in recording data.
    """
    for i in range(len(input_images) // dp_per_record):
        input_image = input_images[dp_per_record * i: (i + 1) * dp_per_record]
        target_image = target_images[dp_per_record * i: (i + 1) * dp_per_record]

        if weight_mask_images is not None:
            weight_mask_image = weight_mask_images[dp_per_record * i: (i + 1) * dp_per_record]
        else:
            weight_mask_image = None

        tfrecord_name = SAVE_FORM.format(prefix, i)

        record_sgm_train_data(
            input_images=input_image,
            target_images=target_image,
            weight_mask_images=weight_mask_image,
            tfrecord_path=tfrecord_name,
            sess=sess
        )
