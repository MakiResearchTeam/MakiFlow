from abc import abstractmethod
import tensorflow as tf
from .tfr_map_method import TFRMapMethod
from .gen_base import GenLayer, PathGenerator


class TFRPathGenerator(PathGenerator):
    TFRECORD = 'tfrecord'

    @abstractmethod
    def next_element(self) -> dict:
        pass


class InputGenLayerV2(GenLayer):
    def __init__(
            self, prefetch_size, batch_size, input_data_type: str,
            tfr_path_generator: TFRPathGenerator, name,
            map_operation: TFRMapMethod, num_parallel_calls=None,
            cycle_length=1, block_length=1,
            shuffle=False, buffer_size=512
    ):
        """

        Parameters
        ----------
        prefetch_size : int
            Number of batches to prepare before feeding into the network.
        batch_size : int
            The batch size.
        input_data_type : str
            Name of the data that is being fed into the network. You have to use iterator classes
            provided for each model in order to refer to the necessary data type.
            For SSD - SSDIterator.IMAGE.
            For NeuralRenderer - NNRIterator.UVMAP.
            For Segmentator - SegmentIterator.IMAGE.
        tfr_path_generator : PathGenerator
            The path generator.
        name : str
            Name of the input layer of the model. You can find it in the
            architecture file.
        map_operation : MapMethod
            Method for mapping paths to the actual data.
        num_parallel_calls : int
            Represents the number of elements to process asynchronously in parallel.
            If not specified, elements will be processed sequentially.
        cycle_length : int
            Controls the number of input elements that are processed concurrently.
        block_length : int
            Each tfrecord in the dataset will be read by blocks of length `block_length`.
        shuffle : bool
            Set to False if you don't want to do shuffling during reading and sampling the data.
        buffer_size: int
            A scalar, representing the number of elements from this dataset from which the new dataset will sample.
            Perfect shuffling is done by setting `buffer_size` to the size of the dataset itself, but it
            requires a buffer which has a size of the dataset.
        """
        self.prefetch_size = prefetch_size
        self.batch_size = batch_size
        self.cycle_length = cycle_length
        self.block_length = block_length
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.iterator = self.build_iterator(tfr_path_generator, map_operation, num_parallel_calls)
        super().__init__(
            name=name,
            input_tensor=self.iterator[input_data_type]
        )

    def build_iterator(self, gen: TFRPathGenerator, map_operation: TFRMapMethod, num_parallel_calls):
        tf_records = tf.data.Dataset.from_generator(
            gen.next_element,
            output_types={
                TFRPathGenerator.TFRECORD: tf.string
            }
        )
        dataset = tf_records.interleave(
            map_func=lambda x: tf.data.TFRecordDataset(x[TFRPathGenerator.TFRECORD]),
            cycle_length=self.cycle_length,
            block_length=self.block_length,
        )

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        dataset = dataset.map(map_func=map_operation.read_record, num_parallel_calls=num_parallel_calls)
        # Set `drop_remainder` to True since otherwise the batch dimension
        # would be None. Example: [None, 1024, 1024, 3]
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self.prefetch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def get_iterator(self):
        return self.iterator
