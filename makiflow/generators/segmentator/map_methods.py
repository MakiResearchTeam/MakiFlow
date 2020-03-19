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

import tensorflow as tf
from makiflow.generators.segmentator.pathgenerator import SegmentPathGenerator
from makiflow.generators.pipeline.map_method import MapMethod, PostMapMethod


class SegmentIterator:
    IMAGE = 'image'
    MASK = 'mask'
    NUM_POSITIVES = 'num_positives'


class LoadResizeNormalize(MapMethod):
    def __init__(
            self, image_shape, mask_shape,
            normalize=255., image_size=None, mask_size=None,
            squeeze_mask=False, calc_positives=False):
        self.image_shape = image_shape
        self.mask_shape = mask_shape
        self.image_size = image_size
        self.mask_size = mask_size
        self.squeeze_mask = squeeze_mask
        self.normalize = tf.constant(normalize, dtype=tf.float32)
        self.calc_positives = calc_positives

    def load_data(self, data_paths):
        img_file = tf.read_file(data_paths[SegmentPathGenerator.IMAGE])
        mask_file = tf.read_file(data_paths[SegmentPathGenerator.MASK])

        img = tf.image.decode_image(img_file)
        mask = tf.image.decode_image(mask_file)

        img.set_shape(self.image_shape)
        mask.set_shape(self.mask_shape)

        if self.squeeze_mask:
            mask = mask[:, :, 0]

        if self.image_size is not None:
            img = tf.image.resize(images=img, size=self.image_size, method=tf.image.ResizeMethod.BILINEAR)

        if self.mask_size is not None:
            mask = tf.image.resize(images=mask, size=self.mask_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img = tf.cast(img, dtype=tf.float32)

        if self.normalize is not None:
            img = tf.divide(img, self.normalize)

        mask = tf.cast(mask, dtype=tf.int32)
        return {
            SegmentIterator.IMAGE: img,
            SegmentIterator.MASK: mask
        }


class LoadDataMethod(MapMethod):
    def __init__(self, image_shape, mask_shape):
        """
        The base map method. Simply loads the data and assigns shapes to it.
        Images are loaded in the RGB format.
        Warning! Shape must be specified according to the actual image (mask) shapes!
        Otherwise set it to [None, None, None].
        Parameters
        ----------
        image_shape : list
            [image width, image height, channels].
        mask_shape : list
            [mask width, mask height, channels].
        """
        self.image_shape = image_shape
        self.mask_shape = mask_shape

    def load_data(self, data_paths):
        img_file = tf.read_file(data_paths[SegmentIterator.IMAGE])
        mask_file = tf.read_file(data_paths[SegmentIterator.MASK])

        img = tf.image.decode_image(img_file)
        mask = tf.image.decode_image(mask_file)

        img.set_shape(self.image_shape)
        mask.set_shape(self.mask_shape)

        img = tf.cast(img, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.int32)
        return {
            SegmentIterator.IMAGE: img,
            SegmentIterator.MASK: mask
        }


class ResizePostMethod(PostMapMethod):
    def __init__(self, image_size=None, mask_size=None, image_resize_method=tf.image.ResizeMethod.BILINEAR,
                 mask_resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
        """
        Resizes the image and the mask accordingly to `image_size` and `mask_size`.
        Parameters
        ----------
        image_size : list
            List of 2 ints: [image width, image height].
        mask_size : list
            List of 2 ints: [mask width, mask height].
        image_resize_method : tf.image.ResizeMethod
            Please refer to the TensorFlow documentation for additional info.
        mask_resize_method : tf.image.ResizeMethod
            Please refer to the TensorFlow documentation for additional info.
        """
        super().__init__()
        self.image_size = image_size
        self.mask_size = mask_size
        self.image_resize_method = image_resize_method
        self.mask_resize_method = mask_resize_method

    def load_data(self, data_paths):
        element = self._parent_method.load_data(data_paths)
        img = element[SegmentIterator.IMAGE]
        mask = element[SegmentIterator.MASK]

        if self.image_size is not None:
            img = tf.image.resize(images=img, size=self.image_size, method=self.image_resize_method)

        if self.mask_size is not None:
            mask = tf.image.resize(images=mask, size=self.mask_size, method=self.mask_resize_method)

        return {
            SegmentIterator.IMAGE: img,
            SegmentIterator.MASK: mask
        }


class NormalizePostMethod(PostMapMethod):
    def __init__(self, divider=255, use_float64=False):
        """
        Normalizes the image by dividing it by the `divider`.
        Parameters
        ----------
        divider : float or int
            The number to divide the image by.
        use_float64 : bool
            Set to True if you want the image to be converted to float64 during normalization.
            It is used for getting more accurate division result during normalization.
        """
        super().__init__()
        self.use_float64 = use_float64
        if use_float64:
            self.divider = tf.constant(divider, dtype=tf.float64)
        else:
            self.divider = tf.constant(divider, dtype=tf.float32)

    def load_data(self, data_paths):
        element = self._parent_method.load_data(data_paths)
        img = element[SegmentIterator.IMAGE]

        if self.use_float64:
            img = tf.cast(img, dtype=tf.float64)
            img = tf.divide(img, self.divider, name='normalizing_image')
            img = tf.cast(img, dtype=tf.float32)
        else:
            img = tf.divide(img, self.divider, name='normalizing_image')

        element[SegmentIterator.IMAGE] = img
        return element


class SqueezeMaskPostMethod(PostMapMethod):
    def __init__(self):
        """
        Use this method if the mask has more than one channel.
        [w, h, c] -> [w, h].
        """
        super().__init__()

    def load_data(self, data_paths):
        element = self._parent_method.load_data(data_paths)
        mask = element[SegmentIterator.MASK]
        mask = mask[:, :, 0]
        element[SegmentIterator.MASK] = mask
        return element


class ComputePositivesPostMethod(PostMapMethod):
    def __init__(self, background_class=0, dtype=tf.int32):
        """
        Computes number of positive samples in the mask.
        Parameters
        ----------
        background_class : int
            Index of the negative class.
        dtype : tf.dtype
            Dtype of the `background_class`. Set it to dtype of the mask.
            Usually mask has dtype of tf.int32. If it's something else just find it out
            trying different dtypes.
        """
        super().__init__()
        self.background = tf.constant(background_class, dtype=dtype)

    def load_data(self, data_paths):
        element = self._parent_method.load_data(data_paths)

        mask = element[SegmentIterator.MASK]
        mask_shape = mask.get_shape().as_list()
        area = mask_shape[0] * mask_shape[1]  # tf.int32
        num_neg = tf.reduce_sum(tf.cast(tf.equal(mask, self.background), dtype=tf.int32))

        num_positives = tf.cast(area - num_neg, dtype=tf.float32)

        element[SegmentIterator.NUM_POSITIVES] = num_positives
        return element


class RGB2BGRPostMethod(PostMapMethod):
    def __init__(self):
        """
        Used for swapping color channels in images from RGB to BGR format.
        """
        super().__init__()

    def load_data(self, data_paths):
        element = self._parent_method.load_data(data_paths)

        img = element[SegmentIterator.IMAGE]
        # Swap channels
        element[SegmentIterator.IMAGE] = tf.reverse(img, axis=[-1], name='RGB2BGR')
        return element


