from .gen_base import  PathGenerator, SegmentIterator

from glob import glob
import os
import numpy as np
from sklearn.utils import shuffle

class CyclicGenerator(PathGenerator):
    def __init__(self, path_images, path_masks):
        self.images = glob(os.path.join(path_images, '*.bmp'))
        self.masks = glob(os.path.join(path_masks, '*.bmp'))

    def next_element(self):
        index = 0
        while True:
            if index % len(self.images) == 0:
                self.images, self.masks = shuffle(self.images, self.masks)
                index = 0

            el = {
                SegmentIterator.image: self.images[index],
                SegmentIterator.mask: self.masks[index]
            }
            index += 1

            yield el

class RandomGenerator(PathGenerator):
    def __init__(self, path_images, path_masks):
        self.images = glob(os.path.join(path_images, '*.bmp'))
        self.masks = glob(os.path.join(path_masks, '*.bmp'))

    def next_element(self):
        while True:
            index = np.random.randint(low=0, high=len(self.images))

            el = {
                SegmentIterator.image: self.images[index],
                SegmentIterator.mask: self.masks[index]
            }

            yield el


class SubCyclicGenerator(PathGenerator):
    def __init__(self, path_batches_images, path_batches_masks):
        assert (len(path_batches_images) == len(path_batches_masks))

        self.batches_images = []
        self.batches_masks = []
        for i in range(len(path_batches_images)):
            self.batches_images.append(glob(os.path.join(path_batches_images[i], '*.bmp')))
            self.batches_masks.append(glob(os.path.join(path_batches_masks[i], '*.bmp')))

    def next_element(self):
        current_batch = 0
        counter_batches = [0 for _ in range(len(self.batches_images))]
        while True:
            if current_batch == len(self.batches_images) and counter_batches[-1] == len(self.batches_images[-1]):
                self.batches_images, self.batches_masks = shuffle(self.batches_images, self.batches_masks)

                self.batches_images = [shuffle(elem) for elem in self.batches_images]
                self.batches_masks = [shuffle(elem) for elem in self.batches_masks]

                current_batch = 0
                counter_batches = [0 for _ in range(len(self.batches_images))]

            el = {
                SegmentIterator.image: self.batches_images[current_batch][counter_batches[current_batch]],
                SegmentIterator.mask: self.batches_masks[current_batch][counter_batches[current_batch]]
            }
            counter_batches[current_batch] += 1
            current_batch += 1

            yield el