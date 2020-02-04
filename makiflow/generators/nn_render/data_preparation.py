from __future__ import absolute_import
import tensorflow as tf
from makiflow.generators.pipeline.tfr.utils import _tensor_to_byte_feature

# Feature names
IMAGE_FNAME = 'IMAGE'
UVMAP_FNAME = 'LOC_MASK'


# Serialize Object Detection Data Point
def serialize_nnr_data_point(image, uvmap):
    feature = {
        IMAGE_FNAME: _tensor_to_byte_feature(image),
        UVMAP_FNAME: _tensor_to_byte_feature(uvmap)
    }
    features = tf.train.Features(feature=feature)
    example_proto = tf.train.Example(features=features)
    return example_proto.SerializeToString()


def record_nnr_train_data(images, uvmaps, tfrecord_path):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for image, loc_mask in zip(images, uvmaps):
            serialized_data_point = serialize_nnr_data_point(image, loc_mask)
            writer.write(serialized_data_point)


# Record data into multiple tfrecords
def record_mp_nnr_train_data(images, uvmaps, prefix, dp_per_record):
    """
    Creates tfrecord dataset where each tfrecord contains `dp_per_second` data points.
    Parameters
    ----------
    images : list or ndarray
        Array of input images.
    uvmaps : list or ndarray
        Array of uvmaps.
    prefix : str
        Prefix for the tfrecords' names. All the filenames will have the same naming pattern:
        `prefix`_`tfrecord index`.tfrecord
    dp_per_record : int
        Data point per tfrecord. Defines how many images (locs, loc_masks, labels) will be
        put into one tfrecord file. It's better to use such `dp_per_record` that
        yields tfrecords of size 300-200 megabytes.
    """
    for i in range(len(images) // dp_per_record):
        image_batch = images[dp_per_record*i: (i+1)*dp_per_record]
        loc_mask_batch = uvmaps[dp_per_record * i: (i + 1) * dp_per_record]
        tfrecord_name = f'{prefix}_{i}.tfrecord'
        record_nnr_train_data(
            images=image_batch,
            uvmaps=loc_mask_batch,
            tfrecord_path=tfrecord_name
        )
