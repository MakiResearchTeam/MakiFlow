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


CLASS_99_MAP_TO = 10 # 13


class BinaryMaskReader:
    def __init__(self, n_classes: int, class_priority=None, class_id_offset=1, class_ind_offset=0, use_image_shape=True):
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
        class_priority : arraylike, optional
            Used to merge binary masks into a single-channel multiclass mask.
            First comes class with the highest priority.
        class_id_offset : int
            Equal to the minimal class id that can be encountered in a mask folder. Default: 1.
        class_ind_offset : int
            Used during masks aggregation (creation of a multiclass flat mask). The class ind in the flat mask
            is calculated as class_priority[i] + class_ind_offset.
            If the minimal class id that can be encountered in a mask folder is 0, then set it to 1.
            If the minimal class id that can be encountered in a mask folder is 1, then set it to 0.
        use_image_shape : bool
            Whether to use shape of an image to determine the shape of its mask.

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
        self.class_priority = class_priority
        self.class_id_offset = class_id_offset
        self.class_ind_offset = class_ind_offset
        self.use_image_shape = use_image_shape
        self.path_generator = None
        self.image_cache = {}
        self.mask_cache = {}

    def load_image(self, path):
        image = self.image_cache.get(path)
        if image is not None:
            return image

        image = cv2.imread(path)
        assert image is not None, f'Could not load image with path={path}.'
        image = image.astype('float32')
        self.image_cache[path] = image
        return image

    def load_mask(self, folder_path, mask_shape=None):
        mask = self.mask_cache.get(folder_path)
        if mask is not None:
            return mask

        # Load individual binary masks into a tensor of masks
        label_tensor = None

        for binary_mask_path in glob(join(folder_path, '*')):
            filename = binary_mask_path.split('/')[-1]
            class_id = int(filename.split('.')[0])
            binary_mask = cv2.imread(binary_mask_path)
            assert binary_mask is not None, f'Could not load mask with path={binary_mask_path}'
            assert class_id - self.class_id_offset >= 0, f'Found a mask with class_id={class_id} that is less than' \
                                                         f'class_id_offset={self.class_id_offset}. Make sure to set ' \
                                                         f'class_id_offset to the minimal possible class_id that can ' \
                                                         f'be found in a folder for a mask.'
            if class_id == 99:
                indx = CLASS_99_MAP_TO
            else:
                indx = class_id - self.class_id_offset

            if label_tensor is None:
                label_tensor = np.zeros(shape=(*binary_mask.shape[:-1], self.n_classes), dtype='int32')

            label_tensor[..., indx] = binary_mask[..., 0]

        # Merge all binary masks into a single-layer multiclass mask if needed
        if self.class_priority:
            label_tensor = self.aggregate_merge(label_tensor)

        self.mask_cache[folder_path] = label_tensor
        return label_tensor

    def aggregate_merge(self, masks) -> np.ndarray:
        assert len(masks) != 0, "Number of masks - 0. Something went wrong"
        final_mask = np.zeros(shape=masks.shape[:-1], dtype='int32')
        # Start with the lowest priority class
        for class_ind in reversed(self.class_priority):
            if class_ind == 99:
                indx = CLASS_99_MAP_TO
            else:
                indx = class_ind - self.class_id_offset
                class_ind += self.class_ind_offset
            layer = masks[..., indx]
            untouched_area = (layer == 0).astype('int32')
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

        mask_shape = None
        if self.use_image_shape:
            mask_shape = image.shape[:2]
        mask = self.load_mask(mask_folder_path, mask_shape=mask_shape)

        return {
            SegmentPathGenerator.IMAGE: image,
            SegmentPathGenerator.MASK: mask
        }

    def __iter__(self):
        assert self.path_generator, 'Path generator has not been provided.'
        return self


class ImageMaskCrop:
    def __init__(self, crop_size):
        """
        Crops image and mask to a specified `crop_size`.

        Parameters
        ----------
        crop_size : tuple
            The crop size (h, w).
        """
        self.crop_h, self.crop_w = crop_size
        self.gen = None

    def __call__(self, gen):
        self.gen = gen
        return self

    def crop(self, image, mask):
        """
        Crops image and mask randomly. The crop position is guarantied to be the same for the mask and the image.

        Parameters
        ----------
        image : np.ndarray
            The image to crop.
        mask : np.ndarray
            The mask to crop

        Returns
        -------
        np.ndarray
            Cropped image.
        np.ndarray
            Cropped mask.
        """
        h, w = image.shape[:2]
        assert h > self.crop_h, f'Height of the received image (h={h}) is larger than the height ' \
                                f'of the crop (crop_h={self.crop_h}).'
        assert h > self.crop_w, f'Width of the received image (h={w}) is larger than the Width ' \
                                f'of the crop (crop_w={self.crop_w}).'

        x = np.random.randint(low=0, high=w - self.crop_w)
        y = np.random.randint(low=0, high=h - self.crop_h)

        image_crop = image[y: y + self.crop_h, x: x + self.crop_w]
        mask_crop = mask[y: y + self.crop_h, x: x + self.crop_w]
        return image_crop, mask_crop

    def __next__(self):
        assert self.gen is not None, 'Generator has not been provided.'
        im_mask = next(self.gen)
        image = im_mask[SegmentPathGenerator.IMAGE]
        mask = im_mask[SegmentPathGenerator.MASK]

        image_crop, mask_crop = self.crop(image, mask)

        im_mask[SegmentPathGenerator.IMAGE] = image_crop
        im_mask[SegmentPathGenerator.MASK] = mask_crop
        return im_mask

    def __iter__(self):
        return self
