from __future__ import absolute_import
import tensorflow as tf
from makiflow.generators.gen_base import TFRMapMethod, SSDIterator
from makiflow.generators.ssd_model.data_preparation import IMAGE_FNAME, LABEL_FNAME, LOC_FNAME, LOC_MASK_FNAME


class LoadDataMethod(TFRMapMethod):
    def __init__(
            self,
            image_dtype=tf.float32,
            label_dtype=tf.int32,
            loc_dtype=tf.float32,
            loc_mask_dtype=tf.float32
    ):
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

        image = tf.io.parse_tensor(example[IMAGE_FNAME], out_type=self.image_dtype)
        label = tf.io.parse_tensor(example[LABEL_FNAME], out_type=self.label_dtype)
        loc = tf.io.parse_tensor(example[LOC_FNAME], out_type=self.loc_dtype)
        loc_mask = tf.io.parse_tensor(example[LOC_MASK_FNAME], out_type=self.loc_mask_dtype)

        return {
            SSDIterator.image: image,
            SSDIterator.loc: loc,
            SSDIterator.loc_mask: loc_mask,
            SSDIterator.label: label
        }


