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
import tensorflow as tf
from .data_preparation import IMAGE_FNAME


class SSPIterator:
    IMAGE = 'IMAGE'


class LoadDataMethod(TFRMapMethod):
    def __init__(
            self,
            image_shape,
            label_shape,
            image_dtype=tf.float32,
            uvmap_dtype=tf.float32
    ):
        self.image_shape = image_shape
        self.uvmap_shape = label_shape

        self.image_dtype = image_dtype
        self.label_dtype = uvmap_dtype

    def read_record(self, serialized_example):
        render_feature_description = {
            IMAGE_FNAME: tf.io.FixedLenFeature((), tf.string),
            UVMAP_FNAME: tf.io.FixedLenFeature((), tf.string),
        }

        example = tf.io.parse_single_example(serialized_example, render_feature_description)

        # Extract the data from the example
        image = tf.io.parse_tensor(example[IMAGE_FNAME], out_type=self.image_dtype)
        uvmap = tf.io.parse_tensor(example[UVMAP_FNAME], out_type=self.label_dtype)

        # Give the data its shape because it doesn't have it right after being extracted
        image.set_shape(self.image_shape)
        uvmap.set_shape(self.uvmap_shape)

        return {
            SSPIterator.IMAGE: image,
            NNRIterator.UVMAP: uvmap
        }