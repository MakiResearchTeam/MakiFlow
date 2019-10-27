from __future__ import absolute_import
from makiflow.base import MakiTensor
import tensorflow as tf
from makiflow.models.segmentation.gen_api import SegmentationGenerator, MapMethod


class GenTrainLayer(MakiTensor):
    def __init__(
            self, prefetch_size, batch_size, generator: SegmentationGenerator, name,
            map_operation: MapMethod
    ):
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

    def build_iterator(self, gen: SegmentationGenerator, map_operation: MapMethod):
        dataset = tf.data.Dataset.from_generator(
            gen.next_element,
            output_types={
                SegmentationGenerator.image: tf.string,
                SegmentationGenerator.mask: tf.string
            }
        )

        dataset = dataset.map(map_operation.load_data)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_size)
        iterator = dataset.make_one_shot_iterator()

        element = iterator.get_next()
        return element[MapMethod.image], element[MapMethod.mask]

    def get_shape(self):
        return self.image.get_shape().to_list()

    def get_name(self):
        return self._name

    def get_params(self):
        return []

    def get_params_dict(self):
        return {}


class GenTrainNPLayer(MakiTensor):
    def __init__(
            self, prefetch_size, batch_size, generator: SegmentationGenerator, name,
            map_operation: MapMethod
    ):
        self.prefetch_size = prefetch_size
        self.batch_size = batch_size
        self.image, self.mask, self.num_positives = self.build_iterator(generator, map_operation)
        self._name = name
        super().__init__(
            data_tensor=self.image,
            parent_layer=self,
            parent_tensor_names=None,
            previous_tensors={}
        )

    def build_iterator(self, gen: SegmentationGenerator, map_operation: MapMethod):
        dataset = tf.data.Dataset.from_generator(
            gen.next_element,
            output_types={
                SegmentationGenerator.image: tf.string,
                SegmentationGenerator.mask: tf.string
            }
        )

        dataset = dataset.map(map_operation.load_data)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_size)
        iterator = dataset.make_one_shot_iterator()

        element = iterator.get_next()
        return element[MapMethod.image], element[MapMethod.mask]

    def get_shape(self):
        return self.image.get_shape().as_list()

    def get_name(self):
        return self._name

    def get_params(self):
        return []

    def get_params_dict(self):
        return {}
