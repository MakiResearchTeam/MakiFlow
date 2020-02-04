from __future__ import absolute_import
import tensorflow as tf
from makiflow.generators.main_modules.utils import _tensor_to_byte_feature

# Feature names
IMAGE_FNAME = 'IMAGE'

# We need to save features' shapes because after reading them from the tfrecord,
# we get Tensors with `unknown` shape.
# It can yield an error if some operations involve access to the shape of the tensors.
IMAGE_H_FNAME = 'IMAGE_HEIGHT'
IMAGE_W_FNAME = 'IMAGE_WIDTH'
IMAGE_C_FNAME = 'IMAGE_CHANNELS'

LOC_MASK_FNAME = 'LOC_MASK'
LOC_MASK_LEN_FNAME = 'LOC_MASK_LEN'

LOC_FNAME = 'LOC'
LOC_LEN_FNAME = 'LOC_LEN'

LABEL_FNAME = 'LABEL'
LABEL_LEN_FNAME = 'LABEL_LEN'


# Serialize Object Detection Data Point
def serialize_od_data_point(image, loc_mask, loc, label):
    feature = {
        IMAGE_FNAME: _tensor_to_byte_feature(image),
        LOC_MASK_FNAME: _tensor_to_byte_feature(loc_mask),
        LOC_FNAME: _tensor_to_byte_feature(loc),
        LABEL_FNAME: _tensor_to_byte_feature(label)
    }
    features = tf.train.Features(feature=feature)
    example_proto = tf.train.Example(features=features)
    return example_proto.SerializeToString()


def record_od_train_data(images, loc_masks, locs, labels, tfrecord_path):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for image, loc_mask, loc, label in zip(images, loc_masks, locs, labels):
            serialized_data_point = serialize_od_data_point(image, loc_mask, loc, label)
            writer.write(serialized_data_point)


# Record data into multiple tfrecords
def record_mp_od_train_data(images, loc_masks, locs, labels, prefix, dp_per_record):
    """
    Creates tfrecord dataset where each tfrecord contains `dp_per_second` data points.
    Parameters
    ----------
    images : list or ndarray
        Array of input images.
    loc_masks : list or ndarray
        Array of localization vector masks for positive and negative samples.
    locs : list or ndarray
        Array of localization vectors for each bounding box.
    labels : list or ndarray
        Array of the label vectors.
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
        loc_mask_batch = loc_masks[dp_per_record*i: (i+1)*dp_per_record]
        loc_batch = locs[dp_per_record*i: (i+1)*dp_per_record]
        label_batch = labels[dp_per_record*i: (i+1)*dp_per_record]
        tfrecord_name = f'{prefix}_{i}.tfrecord'
        record_od_train_data(
            images=image_batch,
            loc_masks=loc_mask_batch,
            locs=loc_batch,
            labels=label_batch,
            tfrecord_path=tfrecord_name
        )

