import numpy as np
from makiflow.generators.main_modules.gen_base import PathGenerator


class RandomGeneratorSegment(PathGenerator):
    def __init__(self, tfrecords):
        """
        Generator for the SSD pipeline, which gives next tfrecord in random order.

        Parameters
        ----------
        tfrecords : list
            List of paths to the tfrecords.
        """
        self._tfrecords = tfrecords

    def next_element(self):
        while True:
            index = np.random.randint(low=0, high=len(self._tfrecords))

            el = {
                TFRPathGenerator.TFRECORD: self._tfrecords[index],
            }

            yield el
