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

from makiflow.generators.pipeline.tfr.tfr_map_method import TFRMapMethod, TFRPostMapMethod
from .data_preparation import INPUT_IMAGE_FNAME, TARGET_IMAGE_FNAME
import tensorflow as tf


class SGMIterator:
    INPUT_IMAGE = 'INPUT_IMAGE'
    TARGET_IMAGE = 'TARGET_IMAGE'


class LoadDataMethod(TFRMapMethod):
    def __init__(
            self,
            input_image_shape,
            target_image_shape,
            input_image_dtype=tf.float32,
            target_image_dtype=tf.int32
    ):
        self.input_image_shape = input_image_shape
        self.target_image_shape = target_image_shape

        self.input_image_dtype = input_image_dtype
        self.target_image_dtype = target_image_dtype

    def read_record(self, serialized_example):
        ssd_feature_description = {
            INPUT_IMAGE_FNAME: tf.io.FixedLenFeature((), tf.string),
            TARGET_IMAGE_FNAME: tf.io.FixedLenFeature((), tf.string),
        }

        example = tf.io.parse_single_example(serialized_example, ssd_feature_description)

        # Extract the data from the example
        input_image = tf.io.parse_tensor(example[INPUT_IMAGE_FNAME], out_type=self.input_image_dtype)
        target_image = tf.io.parse_tensor(example[TARGET_IMAGE_FNAME], out_type=self.target_image_dtype)

        # Give the data its shape because it doesn't have it right after being extracted
        input_image.set_shape(self.input_image_shape)
        target_image.set_shape(self.target_image_shape)

        return {
            SGMIterator.INPUT_IMAGE: input_image,
            SGMIterator.TARGET_IMAGE: target_image
        }


class NormalizePostMethod(TFRPostMapMethod):

    NORMALIZE_TARGET_IMAGE = 'normalize_target_image'
    NORMALIZE_INPUT_IMAGE = 'normalize_input_image'

    def __init__(self, divider=127.5,
                 use_caffee_norm=True,
                 use_float64=True,
                 using_for_input_image=False,
                 using_for_input_image_only=False):
        """
        Normalizes the image by dividing it by the `divider`.
        Parameters
        ----------
        divider : float or int
            The number to divide the image by.
        use_float64 : bool
            Set to True if you want the image to be converted to float64 during normalization.
            It is used for getting more accurate division result during normalization.
        using_for_input_image : bool
            If true, divider will be used on images for generator.
        """
        super().__init__()
        self.use_float64 = use_float64
        self.use_caffe_norm = use_caffee_norm
        self.using_for_input_image = using_for_input_image
        self.using_for_input_image_only = using_for_input_image_only
        if use_float64:
            self.divider = tf.constant(divider, dtype=tf.float64)
        else:
            self.divider = tf.constant(divider, dtype=tf.float32)

    def read_record(self, serialized_example) -> dict:
        element = self._parent_method.read_record(serialized_example)
        target = element[SGMIterator.TARGET_IMAGE]
        if not self.using_for_input_image_only:
            if self.use_float64:
                target = tf.cast(target, dtype=tf.float64)
                if self.use_caffe_norm:
                    target = (target - self.divider) / self.divider
                else:
                    target = tf.divide(target, self.divider, name=NormalizePostMethod.NORMALIZE_TARGET_IMAGE)
                target = tf.cast(target, dtype=tf.float32)
            else:
                if self.use_caffe_norm:
                    target = (target - self.divider) / self.divider
                else:
                    target = tf.divide(target, self.divider, name=NormalizePostMethod.NORMALIZE_TARGET_IMAGE)
            element[SGMIterator.TARGET_IMAGE] = target

        if self.using_for_input_image:
            input_image = element[SGMIterator.INPUT_IMAGE]
            if self.use_float64:
                input_image = tf.cast(input_image, dtype=tf.float64)
                if self.use_caffe_norm:
                    input_image = (input_image - self.divider) / self.divider
                else:
                    input_image = tf.divide(input_image, self.divider, name=NormalizePostMethod.NORMALIZE_INPUT_IMAGE)
                input_image = tf.cast(input_image, dtype=tf.float32)
            else:
                if self.use_caffe_norm:
                    input_image = (input_image - self.divider) / self.divider
                else:
                    input_image = tf.divide(input_image, self.divider, name=NormalizePostMethod.NORMALIZE_INPUT_IMAGE)
            element[SGMIterator.INPUT_IMAGE] = input_image

        return element


class RGB2BGRPostMethod(TFRPostMapMethod):

    RGB2BGR_TARGET_IMAGE = 'RGB2BGR_image'
    RGB2BGR_INPUT_IMAGE = 'BGR2RGB_input'

    def __init__(self, using_for_input_image=False):
        """
        Used for swapping color channels in images from RGB to BGR format.
        Parameters
        ----------
        using_for_input_image : bool
            If true, swapping color channels will be used on input images.
        """
        self.using_for_input_image = using_for_input_image
        super().__init__()

    def read_record(self, serialized_example) -> dict:
        element = self._parent_method.read_record(serialized_example)
        # for image
        target = element[SGMIterator.TARGET_IMAGE]
        # Swap channels
        element[SGMIterator.TARGET_IMAGE] = tf.reverse(target, axis=[-1], name=RGB2BGRPostMethod.RGB2BGR_TARGET_IMAGE)
        # for generator
        if self.using_for_input_image:
            input_image = element[SGMIterator.INPUT_IMAGE]
            # Swap channels
            element[SGMIterator.INPUT_IMAGE] = tf.reverse(input_image, axis=[-1], name=RGB2BGRPostMethod.RGB2BGR_INPUT_IMAGE)
        return element

