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

from .pathgenerator import SegmentPathGenerator
from makiflow.augmentation.segmentation.augment_ops import ElasticAugment
from makiflow.augmentation.segmentation.data_provider import Data
import numpy as np
import cv2


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
