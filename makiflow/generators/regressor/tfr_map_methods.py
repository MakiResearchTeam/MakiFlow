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
from .data_preparation import INPUT_X_FNAME, TARGET_X_FNAME, WEIGHT_MASK_FNAME
import tensorflow as tf


class RIterator:
    INPUT_X = 'INPUT_X'
    TARGET_X = 'TARGET_X'
    WEIGHTS_MASK = 'WEIGHTS_MASK'


class LoadDataMethod(TFRMapMethod):
    def __init__(
            self,
            input_x_shape,
            target_x_shape,
            weight_mask_shape=None,
            input_x_dtype=tf.float32,
            target_x_dtype=tf.float32,
            weight_mask_dtype=tf.float32
    ):
        """
        Method to load data from records

        Parameters
        ----------
        input_x_shape : tuple or list
            Shape of input tensor
        target_x_shape : tuple or list
            Shape of target tensor
        weight_mask_shape : tuple or list
            Shape of weights mask. By default equal to None, i. e. not use in training
        input_x_dtype : tf.dtypes
            Type of target tensor. By default equal to tf.float32
        target_x_dtype : tf.dtypes
            Type of target tensor. By default equal to tf.float32
        weight_mask_dtype : tf.dtypes
            Type of target tensor. By default equal to tf.float32
        """
        self.input_x_shape = input_x_shape
        self.target_x_shape = target_x_shape
        self.weight_mask_shape = weight_mask_shape

        self.input_x_dtype = input_x_dtype
        self.target_x_dtype = target_x_dtype
        self.weight_mask_dtype = weight_mask_dtype

    def read_record(self, serialized_example):
        r_feature_description = {
            INPUT_X_FNAME: tf.io.FixedLenFeature((), tf.string),
            TARGET_X_FNAME: tf.io.FixedLenFeature((), tf.string),
        }

        if self.weight_mask_shape is not None:
            r_feature_description[WEIGHT_MASK_FNAME] = tf.io.FixedLenFeature((), tf.string)

        example = tf.io.parse_single_example(serialized_example, r_feature_description)

        # Extract the data from the example
        input_tensor = tf.io.parse_tensor(example[INPUT_X_FNAME], out_type=self.input_x_dtype)
        target_tensor = tf.io.parse_tensor(example[TARGET_X_FNAME], out_type=self.target_x_dtype)

        if self.weight_mask_shape is not None:
            weights_mask_tensor = tf.io.parse_tensor(example[WEIGHT_MASK_FNAME], out_type=self.weight_mask_dtype)
        else:
            weights_mask_tensor = None

        # Give the data its shape because it doesn't have it right after being extracted
        input_tensor.set_shape(self.input_x_shape)
        target_tensor.set_shape(self.target_x_shape)

        if weights_mask_tensor is not None:
            weights_mask_tensor.set_shape(self.weight_mask_shape)

        output_dict = {
            RIterator.INPUT_X: input_tensor,
            RIterator.TARGET_X: target_tensor
        }

        if weights_mask_tensor is not None:
            output_dict[RIterator.WEIGHTS_MASK] = weights_mask_tensor

        return output_dict


class NormalizePostMethod(TFRPostMapMethod):

    NORMALIZE_TARGET_X = 'normalize_target_tensor'
    NORMALIZE_INPUT_X = 'normalize_input_tensor'

    def __init__(self, divider=127.5,
                 use_caffee_norm=True,
                 use_float64=True,
                 using_for_input_tensor=False,
                 using_for_input_tensor_only=False):
        """
        Normalizes the tensor by dividing it by the `divider`.
        Parameters
        ----------
        divider : float or int
            The number to divide the tensor by.
        use_float64 : bool
            Set to True if you want the tensor to be converted to float64 during normalization.
            It is used for getting more accurate division result during normalization.
        using_for_input_tensor : bool
            If true, divider will be used on tensors for generator.
        """
        super().__init__()
        self.use_float64 = use_float64
        self.use_caffe_norm = use_caffee_norm
        self.using_for_input_tensor = using_for_input_tensor
        self.using_for_input_tensor_only = using_for_input_tensor_only
        if use_float64:
            self.divider = tf.constant(divider, dtype=tf.float64)
        else:
            self.divider = tf.constant(divider, dtype=tf.float32)

    def read_record(self, serialized_example) -> dict:
        element = self._parent_method.read_record(serialized_example)
        target = element[RIterator.TARGET_X]
        if not self.using_for_input_tensor_only:
            if self.use_float64:
                target = tf.cast(target, dtype=tf.float64)
                if self.use_caffe_norm:
                    target = (target - self.divider) / self.divider
                else:
                    target = tf.divide(target, self.divider, name=NormalizePostMethod.NORMALIZE_TARGET_X)
                target = tf.cast(target, dtype=tf.float32)
            else:
                if self.use_caffe_norm:
                    target = (target - self.divider) / self.divider
                else:
                    target = tf.divide(target, self.divider, name=NormalizePostMethod.NORMALIZE_TARGET_X)
            element[RIterator.TARGET_X] = target

        if self.using_for_input_tensor:
            input_tensor = element[RIterator.INPUT_X]
            if self.use_float64:
                input_tensor = tf.cast(input_tensor, dtype=tf.float64)
                if self.use_caffe_norm:
                    input_tensor = (input_tensor - self.divider) / self.divider
                else:
                    input_tensor = tf.divide(input_tensor, self.divider, name=NormalizePostMethod.NORMALIZE_INPUT_X)
                input_tensor = tf.cast(input_tensor, dtype=tf.float32)
            else:
                if self.use_caffe_norm:
                    input_tensor = (input_tensor - self.divider) / self.divider
                else:
                    input_tensor = tf.divide(input_tensor, self.divider, name=NormalizePostMethod.NORMALIZE_INPUT_X)
            element[RIterator.INPUT_X] = input_tensor

        return element


class RGB2BGRPostMethod(TFRPostMapMethod):

    RGB2BGR_TARGET_X = 'RGB2BGR_tensor'
    RGB2BGR_INPUT_X = 'BGR2RGB_input'

    def __init__(self, using_for_input_tensor=False):
        """
        Used for swapping color channels in tensors from RGB to BGR format.
        Parameters
        ----------
        using_for_input_tensor : bool
            If true, swapping color channels will be used on input tensors.
        """
        self.using_for_input_tensor = using_for_input_tensor
        super().__init__()

    def read_record(self, serialized_example) -> dict:
        element = self._parent_method.read_record(serialized_example)
        # for tensor
        target = element[RIterator.TARGET_X]
        # Swap channels
        element[RIterator.TARGET_X] = tf.reverse(target, axis=[-1], name=RGB2BGRPostMethod.RGB2BGR_TARGET_X)
        # for generator
        if self.using_for_input_tensor:
            input_tensor = element[RIterator.INPUT_X]
            # Swap channels
            element[RIterator.INPUT_X] = tf.reverse(input_tensor, axis=[-1], name=RGB2BGRPostMethod.RGB2BGR_INPUT_X)
        return element

