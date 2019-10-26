from __future__ import absolute_import
from makiflow.base import MakiTensor
from abc import abstractmethod
import tensorflow as tf


class SegmentationGenerator(object):
    image = 'image'
    mask = 'mask'
    
    @abstractmethod
    def next_element(self):
        pass


class SegmentGeneratorTrainingLayer(MakiTensor):
    def __init__(
        self, image_shape, mask_shape, prefetch_size, batch_size, generator: SegmentationGenerator, name,
        map_operation=None
        ):
        self.image_shape = image_shape
        self.mask_shape = mask_shape
        self.prefetch_size = prefetch_size
        self.batch_size = batch_size
        self.image, self.mask = self.build_iterator(generator, map_operation)
        self._name = name
        super().__init__(
            data_tensor=self.image,
            parent_layer=self,
            parent_tensor_names=None,
            previous_tensors={}
        )
        
    def build_iterator(self, gen: SegmentationGenerator, map_operation):
        dataset = tf.data.Dataset.from_generator(
            gen.next_element,
            output_types={
                SegmentationGenerator.image: tf.string,
                SegmentationGenerator.mask: tf.string
            }
        )

        if map_operation is not None:
            dataset = dataset.map(map_operation)
        else:
            dataset = dataset.map(self._read_data)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_size)
        iterator = dataset.make_one_shot_iterator()
        
        return iterator.get_next()

    def _read_data(self, data_paths):
        img_file = tf.read_file(data_paths[SegmentationGenerator.image])
        mask_file = tf.read_file(data_paths[SegmentationGenerator.mask])
        
        img = tf.image.decode_image(img_file)
        mask = tf.image.decode_image(mask_file)
        
        img.set_shape(self.image_shape)
        mask.set_shape(self.mask_shape)
        return img, mask

    def get_shape(self):
        return [self.batch_size, *self.image_shape]

    def get_name(self):
        return self._name

    def get_params(self):
        return []

    def get_params_dict(self):
        return {}