from __future__ import absolute_import
import tensorflow as tf
from makiflow.generators.pipeline.tfr.tfr_map_method import TFRMapMethod
from makiflow.generators.ssd.data_preparation import IMAGE_FNAME, LABEL_FNAME, LOC_FNAME, LOC_MASK_FNAME


class SSDIterator:
    LOC = 'loc'
    LOC_MASK = 'loc_mask'
    LABEL = 'label'
    IMAGE = 'IMAGE'


class LoadDataMethod(TFRMapMethod):
    def __init__(
            self,
            image_shape,
            label_shape,
            loc_shape,
            loc_mask_shape,
            image_dtype=tf.float32,
            label_dtype=tf.int32,
            loc_dtype=tf.float32,
            loc_mask_dtype=tf.float32
    ):
        self.image_shape = image_shape
        self.label_shape = label_shape
        self.loc_shape = loc_shape
        self.loc_mask_shape = loc_mask_shape

        self.image_dtype = image_dtype
        self.label_dtype = label_dtype
        self.loc_dtype = loc_dtype
        self.loc_mask_dtype = loc_mask_dtype

    def read_record(self, serialized_example):
        ssd_feature_description = {
            IMAGE_FNAME: tf.io.FixedLenFeature((), tf.string),
            LABEL_FNAME: tf.io.FixedLenFeature((), tf.string),
            LOC_FNAME: tf.io.FixedLenFeature((), tf.string),
            LOC_MASK_FNAME: tf.io.FixedLenFeature((), tf.string)
        }

        example = tf.io.parse_single_example(serialized_example, ssd_feature_description)

        # Extract the data from the example
        image = tf.io.parse_tensor(example[IMAGE_FNAME], out_type=self.image_dtype)
        label = tf.io.parse_tensor(example[LABEL_FNAME], out_type=self.label_dtype)
        loc = tf.io.parse_tensor(example[LOC_FNAME], out_type=self.loc_dtype)
        loc_mask = tf.io.parse_tensor(example[LOC_MASK_FNAME], out_type=self.loc_mask_dtype)

        # Give the data its shape because it doesn't have it right after being extracted
        image.set_shape(self.image_shape)
        label.set_shape(self.label_shape)
        loc.set_shape(self.loc_shape)
        loc_mask.set_shape(self.loc_mask_shape)

        return {
            SSDIterator.IMAGE: image,
            SSDIterator.LOC: loc,
            SSDIterator.LOC_MASK: loc_mask,
            SSDIterator.LABEL: label
        }



