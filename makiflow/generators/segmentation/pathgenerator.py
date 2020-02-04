from abc import ABC

from makiflow.generators.main_modules.gen_base import PathGenerator
from makiflow.generators.segmentation.map_methods import SegmentIterator
from glob import glob
import os
import numpy as np
from sklearn.utils import shuffle


class SegmentPathGenerator(PathGenerator, ABC):
    image = 'image'
    mask = 'mask'


class CyclicGeneratorSegment(SegmentPathGenerator):
    def __init__(self, path_images, path_masks):
        """
        Generator for pipeline, which gives next element in cycle order

        Parameters
        ----------
        path_images : str
            Path to the masks folder. Example: /home/mnt/data/batch_1/masks
        path_masks : str
            Path to the images folder. Example: /home/mnt/data/batch_1/images
        """
        self.images = glob(os.path.join(path_images, '*.bmp'))
        self.masks = glob(os.path.join(path_masks, '*.bmp'))

    def next_element(self):
        index = 0
        while True:
            if index % len(self.images) == 0:
                self.images, self.masks = shuffle(self.images, self.masks)
                index = 0

            el = {
                SegmentIterator.IMAGE: self.images[index],
                SegmentIterator.MASK: self.masks[index]
            }
            index += 1

            yield el


class RandomGeneratorSegment(SegmentPathGenerator):
    def __init__(self, path_images, path_masks):
        """
        Generator for pipeline, which gives next element in random order

        Parameters
        ----------
        path_images : str
            Path to the masks folder. Example: /home/mnt/data/batch_1/masks
        path_masks : str
            Path to the images folder. Example: /home/mnt/data/batch_1/images
        """
        self.images = glob(os.path.join(path_images, '*.bmp'))
        self.masks = glob(os.path.join(path_masks, '*.bmp'))

    def next_element(self):
        while True:
            index = np.random.randint(low=0, high=len(self.images))

            el = {
                SegmentIterator.IMAGE: self.images[index],
                SegmentIterator.MASK: self.masks[index]
            }

            yield el


class SubCyclicGeneratorSegment(SegmentPathGenerator):
    def __init__(self, path_batches_images, path_batches_masks):
        """
        Generator for pipeline, which gives next element in sub-cyclic order
        Parameters
        ----------
        path_batches_masks : list
            A list of groups of paths to masks.
        path_batches_images : list
            A list of groups of paths to images.
        """
        assert (len(path_batches_images) == len(path_batches_masks))

        self.batches_images = path_batches_images
        self.batches_masks = path_batches_masks

        self.batches_images, self.batches_masks = shuffle(self.batches_images, self.batches_masks)

        for i in range(len(self.batches_masks)):
            self.batches_images[i], self.batches_masks[i] = shuffle(self.batches_images[i], self.batches_masks[i])

    def next_element(self):
        current_batch = 0
        counter_batches = [0] * len(self.batches_masks)
        while True:
            if current_batch == (len(self.batches_images) - 1) and counter_batches[-1] == (
                    len(self.batches_images[-1]) - 1):
                self.batches_images, self.batches_masks = shuffle(self.batches_images, self.batches_masks)

                for i in range(len(self.batches_masks)):
                    self.batches_images[i], self.batches_masks[i] = shuffle(self.batches_images[i], self.batches_masks[i])

                current_batch = 0
                counter_batches = [0] * len(self.batches_masks)

            el = {
                SegmentIterator.IMAGE: self.batches_images[current_batch][counter_batches[current_batch]],
                SegmentIterator.MASK: self.batches_masks[current_batch][counter_batches[current_batch]]
            }

            counter_batches[current_batch] += 1
            current_batch = (current_batch + 1) % len(self.batches_images)

            yield el
