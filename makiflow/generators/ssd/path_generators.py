import numpy as np
from makiflow.generators.tfr_gen_base import TFRPathGenerator


class RandomGeneratorSegment(TFRPathGenerator):
    def __init__(self, tfrecords):
        """
        Generator for the SSD pipeline, which gives next tfrecord in random order.

        Parameters
        ----------
        tfrecords : list
            List of paths to the tfrecords.
        """
        self.tfrecords = tfrecords

    def next_element(self):
        while True:
            index = np.random.randint(low=0, high=len(self.tfrecords))

            el = {
                TFRPathGenerator.TFRECORD: self.tfrecords[index],
            }

            yield el
