from __future__ import absolute_import
import tensorflow as tf
from makiflow.generators.ssd_model.utils import _tensor_to_byte_feature

# Feature names
IMAGE_FNAME = 'IMAGE'
LOC_MASK_FNAME = 'LOC_MASK'
LOC_FNAME = 'LOC'
LABEL_FNAME = 'LABEL'


# Serialize Object Detection Data Point
def serialize_od_data_point(image, loc_mask, loc, label):
    feature = {
        'image': _tensor_to_byte_feature(image),
        'loc_mask': _tensor_to_byte_feature(loc_mask),
        'loc': _tensor_to_byte_feature(loc),
        'label': _tensor_to_byte_feature(label)
    }
    features = tf.train.Features(feature=feature)
    example_proto = tf.train.Example(features=features)
    return example_proto.SerializeToString()


def record_od_train_data(images, loc_masks, locs, labels, tfrecord_path):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for image, loc_mask, loc, label in zip(images, loc_masks, locs, labels):
            serialized_data_point = serialize_od_data_point(image, loc_mask, loc, label)
            writer.write(serialized_data_point)
