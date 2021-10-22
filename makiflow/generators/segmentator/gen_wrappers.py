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
import numpy as np
import cv2
from glob import glob
from os.path import join

from .pathgenerator import SegmentPathGenerator
from makiflow.augmentation.segmentation.augment_ops import ElasticAugment


def data_reader_wrapper(gen, use_bgr2rgb=False) -> dict:
    while True:
        path_dict = next(gen)
        image = cv2.imread(path_dict[SegmentPathGenerator.IMAGE])
        if use_bgr2rgb:
            image = image[..., ::-1]

        mask = cv2.cvtColor(
            cv2.imread(path_dict[SegmentPathGenerator.MASK]),
            cv2.COLOR_BGR2GRAY
        )
        yield {
            SegmentPathGenerator.IMAGE: image.astype(np.float32, copy=False),
            SegmentPathGenerator.MASK: mask.astype(np.int32, copy=False)
        }


def data_resize_wrapper(gen, resize_to: tuple):
    """

    Parameters
    ----------
    gen
    resize_to : tuple
        (H, W)

    Returns
    -------
    same dict as input, but with resized images

    """
    while True:
        data_dict = next(gen)
        image, mask = (data_dict[SegmentPathGenerator.IMAGE], data_dict[SegmentPathGenerator.MASK])
        yield {
            SegmentPathGenerator.IMAGE: cv2.resize(
                image,
                (resize_to[1], resize_to[0]),
                interpolation=cv2.INTER_LINEAR
            ).astype(np.float32, copy=False),

            SegmentPathGenerator.MASK: cv2.resize(
                mask.astype(np.float32, copy=False),
                (resize_to[1], resize_to[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32, copy=False)
        }


def data_elastic_wrapper(
        gen, alpha=500, std=8, num_maps=10, noise_invert_scale=5, seed=None,
        img_inter='linear', mask_inter='nearest', border_mode='reflect',
        keep_old_data=True, image_shape=(1024, 1024), prob=0.5, aug_update_period=300
):
    elastic_aug = ElasticAugment(
        alpha=alpha, std=std, num_maps=num_maps, noise_invert_scale=noise_invert_scale,
        seed=seed, img_inter=img_inter, mask_inter=mask_inter, border_mode=border_mode,
        keep_old_data=keep_old_data
    )
    elastic_aug.setup_augmentor(image_shape)

    counter = 1
    while True:
        data_dict = next(gen)
        image, mask = data_dict[SegmentPathGenerator.IMAGE], data_dict[SegmentPathGenerator.MASK]

        if np.random.uniform() < prob:
            image, mask = elastic_aug.perform_augment(image, mask)

        yield {
            SegmentPathGenerator.IMAGE: image,
            SegmentPathGenerator.MASK: mask
        }

        counter += 1
        # Update mapping fields
        if counter % aug_update_period:
            counter = 0
            elastic_aug.setup_augmentor(image_shape)


def binary_masks_reader(gen, n_classes, image_shape):
    image_cache = {}
    label_cache = {}
    while True:
        path_dict = next(gen)

        # Get image
        if image_cache.get(path_dict[SegmentPathGenerator.IMAGE]) is None:
            # Load data
            image = cv2.imread(path_dict[SegmentPathGenerator.IMAGE])
            # Cache data
            image_cache[path_dict[SegmentPathGenerator.IMAGE]] = image.astype(np.float32, copy=False)

        # Get label tensor
        if label_cache.get(path_dict[SegmentPathGenerator.MASK]) is None:
            # Load data
            mask_folder = path_dict[SegmentPathGenerator.MASK]
            label_tensor = np.zeros(shape=(*image_shape, n_classes), dtype='int32')
            for binary_mask_path in glob(join(mask_folder, '*')):
                filename = binary_mask_path.split('/')[-1]
                class_id = int(filename.split('.')[0])
                assert class_id != 0, 'Encountered class 0. Class names must start from 1.'
                binary_mask = cv2.imread(binary_mask_path)
                assert binary_mask is not None, f'Could not load mask with name={binary_mask_path}'
                label_tensor[..., class_id - 1] = binary_mask[..., 0]
            # Cache data
            label_cache[path_dict[SegmentPathGenerator.MASK]] = label_tensor

        yield {
            SegmentPathGenerator.IMAGE: image_cache[path_dict[SegmentPathGenerator.IMAGE]],
            SegmentPathGenerator.MASK: label_cache[path_dict[SegmentPathGenerator.MASK]]
        }


class BinaryMaskReader:
    def __init__(self, n_classes: int, image_shape: tuple, class_priority=None):
        """
        Reads binary masks from the mask folder and aggregates them into a label tensor.
        By default all the masks are being aggregated into a tensor of labels of
        shape (height, width, n_classes).
        If `class_priority` is provided, all the masks are aggregated into a tensor of labels
        of shape (height, width, 1). Pixel value is an index of a class with highest priority that is present
        at that location of the image.
        This wrapper is caching all the masks and images, so be careful when using it on large datasets.

        Parameters
        ----------
        n_classes : int
            Number of classes in the data.
        image_shape : tuple
            (Heights, Width).
        class_priority : arraylike, optional
            Used to merge binary masks into a single-channel multiclass mask.
            First comes class with the highest priority.

        Notes
        -----
        The data must be organized as follows:
        - /images
            - 1.bmp
            - 2.bmp
            - ...
        - /masks
            - /1 - names of an image and its corresponding mask folder must be the same.
                - 1.bmp - name of the mask in this case is the class id.
                - 3.bmp
            - /2
                - 1.bmp
                - 5.bmp
                - 6.bmp
            - ...
        """
        self.n_classes = n_classes
        self.image_shape = image_shape
        self.class_priority = class_priority
        self.path_generator = None
        self.image_cache = {}
        self.mask_cache = {}

    def load_image(self, path):
        image = self.image_cache.get(path)
        if image:
            return image

        image = cv2.imread(path)
        assert image, f'Could not load image with path={path}.'
        image = image.astype('float32')
        self.image_cache[path] = image
        return image

    def load_mask(self, folder_path):
        mask = self.mask_cache.get(folder_path)
        if mask:
            return mask

        # Load individual binary masks into a tensor of masks
        label_tensor = np.zeros(shape=(*self.image_shape, self.n_classes), dtype='int32')
        for binary_mask_path in glob(join(folder_path, '*')):
            filename = binary_mask_path.split('/')[-1]
            class_id = int(filename.split('.')[0])
            assert class_id != 0, 'Encountered class 0. Since 0 is reserved for the background, ' \
                                  'class names (filenames) must start from 1.'
            binary_mask = cv2.imread(binary_mask_path)
            assert binary_mask is not None, f'Could not load mask with path={binary_mask_path}'
            label_tensor[..., class_id - 1] = binary_mask[..., 0]

        # Merge all binary masks into a single-layer multiclass mask if needed
        if self.class_priority:
            label_tensor = self.aggregate_merge(label_tensor)

        self.mask_cache[folder_path] = label_tensor
        return label_tensor

    def aggregate_merge(self, masks):
        final_mask = np.zeros(shape=self.image_shape, dtype='int32')
        # Start with the lowest priority class
        for class_ind in self.class_priority.reverse():
            layer = masks[..., class_ind - 1]
            untouched_area = (layer != 0).astype('int32')
            final_mask = final_mask * untouched_area + layer * class_ind
        return final_mask

    def __call__(self, path_generator):
        self.path_generator = path_generator
        return self

    def __next__(self):
        assert self.path_generator, 'Path generator has not been provided.'
        data_paths = next(self.path_generator)
        image_path = data_paths[SegmentPathGenerator.IMAGE]
        mask_folder_path = data_paths[SegmentPathGenerator.MASK]
        image = self.load_image(image_path)
        mask = self.load_mask(mask_folder_path)
        return {
            SegmentPathGenerator.IMAGE: image,
            SegmentPathGenerator.MASK: mask
        }

    def __iter__(self):
        assert self.path_generator, 'Path generator has not been provided.'
        return self
