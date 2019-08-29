from __future__ import absolute_import
from makiflow.augmentation.segmentation.base import AugmentOp, Augmentor
from scipy.ndimage import gaussian_filter
import numpy as np
import cv2


class ElasticAugment(AugmentOp):
    def __init__(self, alpha=500, std=8, num_maps=10, noise_invert_scale=5, random_state=None, keep_old_data=True):
        super().__init__()
        self.alpha = alpha
        self.std = std
        self.num_maps = num_maps
        self.noise_invert_scale = noise_invert_scale
        self.random_state = np.random.RandomState(random_state)
        self.keep_old_data = keep_old_data

    def _generate_maps(self):
        # List of tuples (xmap, ymap)
        self._maps = []
        for _ in range(self.num_maps):
            dx = gaussian_filter(
                (
                        self.random_state.rand(
                            self._img_shape[0] // self.noise_invert_scale,
                            self._img_shape[1] // self.noise_invert_scale
                        ) * 2 - 1
                ),
                self.std,
                mode='nearest'
            ) * self.alpha
            dy = gaussian_filter(
                (
                        self.random_state.rand(
                            self._img_shape[0] // self.noise_invert_scale,
                            self._img_shape[1] // self.noise_invert_scale
                        ) * 2 - 1
                ),
                self.std,
                mode='nearest'
            ) * self.alpha
            dx = cv2.resize(dx, (self._img_shape[1], self._img_shape[0]))
            dy = cv2.resize(dy, (self._img_shape[1], self._img_shape[0]))

            x, y = np.meshgrid(np.arange(self._img_shape[1]), np.arange(self._img_shape[0]))
            mapx = np.float32(x + dx)
            mapy = np.float32(y + dy)
            self._maps += [mapx, mapy]

    def get_data(self):
        imgs, masks = self._data.get_data()

        new_imgs, new_masks = [], []
        for img, mask in zip(imgs, masks):
            for mapx, mapy in self._maps:
                new_imgs.append(cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101))
                new_masks.append(cv2.remap(mask, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101))

        if self.keep_old_data:
            new_imgs += imgs
            new_masks += masks

        return new_imgs, masks

    def __call__(self, data: Augmentor):
        super().__call__(data)
        self._generate_maps()
        return self
