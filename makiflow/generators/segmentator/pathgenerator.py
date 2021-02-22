# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

from abc import ABC
from makiflow.generators.pipeline.gen_base import PathGenerator
from glob import glob
import os
import numpy as np
from sklearn.utils import shuffle


class SegmentPathGenerator(PathGenerator, ABC):
    IMAGE = 'image'
    MASK = 'mask'


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
                SegmentPathGenerator.IMAGE: self.images[index],
                SegmentPathGenerator.MASK: self.masks[index]
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
                SegmentPathGenerator.IMAGE: self.images[index],
                SegmentPathGenerator.MASK: self.masks[index]
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
                SegmentPathGenerator.IMAGE: self.batches_images[current_batch][counter_batches[current_batch]],
                SegmentPathGenerator.MASK: self.batches_masks[current_batch][counter_batches[current_batch]]
            }

            counter_batches[current_batch] += 1
            current_batch = (current_batch + 1) % len(self.batches_images)

            yield el


class DistributionBasedPathGen(SegmentPathGenerator):
    def __init__(self, image_mask_dict, groupid_image_dict, groupid_prob_dict, update_period=300):
        """
        Yields paths based off of a give distribution.

        Parameters
        ----------
        image_mask_dict : dict
            Contains pairs { image_path: masks_folder_path }.
        groupid_image_dict : dict
            Contains pairs { groupid: [image_path] }, where `groupid` value starts from 0.
            Example: { 0: ['0.bmp', '1.bmp], 1: ['2.bmp', '3.bmp'] }.
        groupid_prob_dict : dict
            Contains pairs { groupid: prob }, where `prob` is a probability of selecting images from
            `groupid` group.
        update_period : int
            Every `update_period` iterations all the lists in `groupid_image_dict` are being shuffled.
        """
        self._groupids = []
        self._groupid_distribution = []
        for groupid, prob in groupid_prob_dict.items():
            self._groupids.append(groupid)
            self._groupid_distribution.append(prob)

        self._image_mask_dict = image_mask_dict
        self._groupid_image_dict = groupid_image_dict
        self._update_period = update_period
        # Being updated in the main loop - `next_element` method
        self._iteration_counter = 0
        self._create_group_path_generators()

    def _create_group_path_generators(self):
        """
        Creates generators for each image group to iterate over images in that group
        indefinitely.
        """
        def make_infinite_gen(image_paths):
            index = 0
            while True:
                image_path = image_paths[index]
                yield {
                    SegmentPathGenerator.IMAGE: image_path,
                    SegmentPathGenerator.MASK: self._image_mask_dict[image_path]
                }

                index += 1

                if index == len(image_paths):
                    index = 0

                if self._iteration_counter % self._update_period:
                    image_paths = shuffle(image_paths)

        self._group_generators = {}
        for group_id, group_images in self._groupid_image_dict.items():
            self._group_generators[group_id] = make_infinite_gen(group_images)

    def next_element(self):
        iteration_counter = 0
        while True:
            groupid = np.random.choice(self._groupids, p=self._groupid_distribution)
            yield next(self._group_generators[groupid])
            iteration_counter += 1
