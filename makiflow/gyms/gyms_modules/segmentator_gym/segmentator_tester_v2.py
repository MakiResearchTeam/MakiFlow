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
import glob
import os
from makiflow.gyms.gyms_modules.gyms_collector import GymCollector, SEGMENTATION, TESTER
from makiflow.gyms.gyms_modules.segmentator_gym import SegmentatorTester
from makiflow.gyms.gyms_modules.segmentator_gym.utils import draw_heatmap
from makiflow.metrics import bin_categorical_dice_coeff
from makiflow.metrics import confusion_mat
import cv2
import numpy as np


class SegmentatorTesterV2(SegmentatorTester):
    """
    The same as SegmentatorTester except that it uses masks in the same format as
    SegmentatorBinaryTester does.
    """
    THREASH_HOLD = 'threash_hold'
    THREASHOLD_DEFAULT = 0.5
    CLASS_PRIORITY = 'class_priority'

    def _init(self):
        self._thr_hold = self._config.get(SegmentatorTesterV2.THREASH_HOLD, SegmentatorTesterV2.THREASHOLD_DEFAULT)

        self._class_priority = self._config.get(SegmentatorTesterV2.CLASS_PRIORITY)
        assert self._class_priority is not None, "class_priority parameter has not" \
                                                 " been provided in the configuration file."
        super()._init()

    def _init_train_images(self):
        if not isinstance(self._config[SegmentatorTesterV2.TRAIN_IMAGE], list):
            train_images_path = [self._config[SegmentatorTesterV2.TRAIN_IMAGE]]
        else:
            train_images_path = self._config[SegmentatorTesterV2.TRAIN_IMAGE]

        self._train_masks_path = self._config[self.TRAIN_MASK]
        self._norm_images_train = []
        self._train_images = []
        self._train_masks_np = []
        self._names_train = []

        for i in range(len(train_images_path)):
            # Image
            norm_img, orig_img = self._preprocess(train_images_path[i])
            self._norm_images_train.append(
                norm_img
            )
            self._train_images.append(orig_img.astype(np.uint8))
            n_images = 2
            # Mask
            if self._train_masks_path is not None and len(self._train_masks_path) > i:
                orig_mask = self._preprocess_masks(self._train_masks_path[i])
                self._train_masks_np.append(orig_mask.astype(np.uint8))
                n_images += orig_mask.shape[-1]

            self._names_train.append(SegmentatorTesterV2.TRAIN_N.format(i))
            self.add_image(self._names_train[-1], n_images=n_images)

    def _init_test_images(self):
        self._test_masks_path = self._config[self.TEST_MASK]
        if not isinstance(self._config[SegmentatorTesterV2.TEST_IMAGE], list):
            test_images_path = [self._config[SegmentatorTesterV2.TEST_IMAGE]]
        else:
            test_images_path = self._config[SegmentatorTesterV2.TEST_IMAGE]

        self._test_norm_images = []
        self._test_images = []
        self._test_mask_np = []
        self._names_test = []

        for i, single_path in enumerate(test_images_path):
            # Image
            norm_img, orig_img = self._preprocess(single_path)
            self._test_norm_images.append(
                norm_img
            )
            self._test_images.append(orig_img.astype(np.uint8))
            n_images = 2
            # Mask
            if self._test_masks_path is not None and len(self._test_masks_path) > i:
                orig_mask = self._preprocess_masks(self._test_masks_path[i])
                self._test_mask_np.append(orig_mask.astype(np.uint8))
                n_images += orig_mask.shape[-1]

            self._names_test.append(SegmentatorTesterV2.TEST_N.format(i))
            # Image + orig mask (if was given) + prediction
            self.add_image(self._names_test[-1], n_images=n_images)
        if self._test_masks_path is not None:
            # Add confuse matrix image
            self._names_test += [self.CONFUSE_MATRIX]
            self.add_image(self._names_test[-1])

    def _preprocess_masks(self, path_mask_folder: str):
        if self._resize_to is not None:
            h, w = self._resize_to
        else:
            first_img = cv2.imread(glob.glob(os.path.join(path_mask_folder, '*.bmp'))[0])
            h, w = first_img.shape[:2]
        n_classes = len(self._config[SegmentatorTester.CLASSES_NAMES])

        labels = np.zeros((h, w, n_classes), dtype='int32')

        for i in range(len(self._config[SegmentatorTester.CLASSES_NAMES])):
            single_label = cv2.imread(os.path.join(path_mask_folder, f'{i+1}.bmp'))
            if single_label is not None:
                _, labels[..., i] = super()._preprocess(single_label, mask_preprocess=True)

        labels = self.__aggregate_merge(labels, (h, w))
        return labels

    def __aggregate_merge(self, masks_tensor, mask_shape):
        final_mask = np.zeros(shape=mask_shape, dtype='int32')
        # Start with the lowest priority class
        for class_ind in reversed(self._class_priority):
            layer = masks_tensor[..., class_ind - 1]
            untouched_area = (layer == 0).astype('int32')
            final_mask = final_mask * untouched_area + layer * class_ind
        return final_mask


GymCollector.update_collector(SEGMENTATION, TESTER, SegmentatorTesterV2)
