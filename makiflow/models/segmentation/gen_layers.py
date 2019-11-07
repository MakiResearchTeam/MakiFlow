from __future__ import absolute_import
import tensorflow as tf
from makiflow.models.segmentation.gen_base import PathGenerator, MapMethod, GenLayer, SegmentIterator


class InputGenLayer(GenLayer):
    def __init__(
            self, prefetch_size, batch_size, generator: PathGenerator, name,
            map_operation: MapMethod
    ):
        self.prefetch_size = prefetch_size
        self.batch_size = batch_size
        self.iterator = self.build_iterator(generator, map_operation)
        super().__init__(
            name=name,
            input_image=self.iterator[SegmentIterator.image]
        )

    def build_iterator(self, gen: PathGenerator, map_operation: MapMethod):
        dataset = tf.data.Dataset.from_generator(
            gen.next_element,
            output_types={
                PathGenerator.image: tf.string,
                PathGenerator.mask: tf.string
            }
        )

        dataset = dataset.map(map_operation.load_data)
        # Set `drop_remainder` to True since otherwise the batch dimension
        # would be None. Example: [None, 1024, 1024, 3]
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self.prefetch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def get_iterator(self):
        return self.iterator
