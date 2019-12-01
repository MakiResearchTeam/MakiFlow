from __future__ import absolute_import
import tensorflow as tf
from makiflow.generators.ssd.utils import _tensor_to_byte_feature

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
