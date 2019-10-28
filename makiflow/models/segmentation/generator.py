from __future__ import absolute_import
import tensorflow as tf
from makiflow.models.segmentation.gen_api import SegmentationGenerator, MapMethod, GenTrainLayer


class GenTrainLayerBasic(GenTrainLayer):
    def __init__(
            self, prefetch_size, batch_size, generator: SegmentationGenerator, name,
            map_operation: MapMethod
    ):
        self.prefetch_size = prefetch_size
        self.batch_size = batch_size
        image, self.mask = self.build_iterator(generator, map_operation)
        super().__init__(
            name=name,
            image=image
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


class GenTrainNPLayer(GenTrainLayer):
    def __init__(
            self, prefetch_size, batch_size, generator: SegmentationGenerator, name,
            map_operation: MapMethod
    ):
        self.prefetch_size = prefetch_size
        self.batch_size = batch_size
        image, self.mask, self.num_positives = self.build_iterator(generator, map_operation)
        super().__init__(
            name=name,
            image=image
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
        return element[MapMethod.image], element[MapMethod.mask], element[MapMethod.num_positives]

