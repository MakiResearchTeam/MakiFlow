from makiflow.generators.pipeline.tfr.tfr_map_method import TFRMapMethod, TFRPostMapMethod
from .data_preparation import IMAGE_FNAME, UVMAP_FNAME
import tensorflow as tf
import numpy as np


class NNRIterator:
    UVMAP = 'UVMAP'
    IMAGE = 'IMAGE'


class LoadDataMethod(TFRMapMethod):
    def __init__(
            self,
            image_shape,
            label_shape,
            image_dtype=tf.float32,
            uvmap_dtype=tf.int32
    ):
        self.image_shape = image_shape
        self.uvmap_shape = label_shape

        self.image_dtype = image_dtype
        self.label_dtype = uvmap_dtype

    def read_record(self, serialized_example):
        ssd_feature_description = {
            IMAGE_FNAME: tf.io.FixedLenFeature((), tf.string),
            UVMAP_FNAME: tf.io.FixedLenFeature((), tf.string),
        }

        example = tf.io.parse_single_example(serialized_example, ssd_feature_description)

        # Extract the data from the example
        image = tf.io.parse_tensor(example[IMAGE_FNAME], out_type=self.image_dtype)
        uvmap = tf.io.parse_tensor(example[UVMAP_FNAME], out_type=self.label_dtype)

        # Give the data its shape because it doesn't have it right after being extracted
        image.set_shape(self.image_shape)
        uvmap.set_shape(self.uvmap_shape)

        return {
            NNRIterator.IMAGE: image,
            NNRIterator.UVMAP: uvmap
        }


class RandomCropMethod(TFRPostMapMethod):
    def __init__(self, crop_w, crop_h):
        """
        Perform random crop of the input images and their corresponding uvmaps.
        Parameters
        ----------
        crop_w : int
            Width of the crop.
        crop_h : int
            Height of the crop.
        """
        super().__init__()
        self._crop_w = crop_w
        self._crop_h = crop_h
        self._image_crop_size = [crop_h, crop_w, 3]
        self._uvmap_crop_size = [crop_h, crop_w, 2]
        self._image_crop_size_tf = tf.constant(np.array([crop_h, crop_w, 3], dtype=np.int32))
        self._uvmap_crop_size_tf = tf.constant(np.array([crop_h, crop_w, 2], dtype=np.int32))

    def read_record(self, serialized_example) -> dict:
        element = self._parent_method.read_record(serialized_example)
        image = element[NNRIterator.IMAGE]
        uvmap = element[NNRIterator.UVMAP]
        # This is an adapted code from the original TensorFlow's `random_crop` method
        limit = tf.shape(image) - self._image_crop_size_tf + 1
        offset = tf.random_uniform(

            shape=[3],
            dtype=tf.int32,
            # it is unlikely that a tensor with shape more that 10000 will appear
            maxval=10000
        ) % limit

        cropped_image = tf.slice(image, offset, self._image_crop_size_tf)
        cropped_uvmap = tf.slice(uvmap, offset, self._uvmap_crop_size_tf)
        # After slicing the tensors doesn't have proper shape. They get instead [None, None, None].
        # We can't use tf.Tensors for setting shape because they are note iterable what causes errors.
        cropped_image.set_shape(self._image_crop_size)
        cropped_uvmap.set_shape(self._uvmap_crop_size)

        element[NNRIterator.IMAGE] = cropped_image
        element[NNRIterator.UVMAP] = cropped_uvmap
        return element
