from __future__ import absolute_import
from makiflow.augmentation.segmentation.base import AugmentOp, Augmentor
from scipy.ndimage import gaussian_filter
import numpy as np
import cv2

INTERPOLATION_TYPE = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'cubic': cv2.INTER_CUBIC
}


class ElasticAugment(AugmentOp):
    def __init__(
            self, alpha=500, std=8, num_maps=10, noise_invert_scale=5, seed=None,
            img_inter='linear', mask_inter='nearest',
            keep_old_data=True
    ):
        """
        Performs elastic transformation.
        Parameters
        ----------
        alpha : int
            Affects curvature.
        std : int
            Affects curvature.
        num_maps : int
            Number of elastic mappings. All the generated maps will be applied to the given images
            and masks, thus there will be `num_maps`*`img_count` new images.
        noise_invert_scale : int
            The noise tensors will be created of size
            (img_w // `noise_invert_scale`, img_h // `noise_invert_scale`).
            For the sake of controlling the amount of elastic deformation it is recommended to create
            noise tensors with less size than the original image and then upscale it to the
            appropriate size.
            Bigger the `noise_invert_scale`, less 'aggressive' the deformation is.
        seed : int (optional)
            Seed for the random generator that will generate noise tensors for the
            elastic transformation maps.
        img_inter : str
            Image interpolation type. Can be 'nearest', 'linear' or 'cubic'.
        mask_inter : str
            Image interpolation type. Can be 'nearest', 'linear' or 'cubic'.
        keep_old_data : bool
            Set to false if you don't want to include unaugmented images into the final data set.
        """
        super().__init__()
        self.alpha = alpha
        self.std = std
        self.num_maps = num_maps
        self.noise_invert_scale = noise_invert_scale
        self.random_state = np.random.RandomState(seed)
        self.keep_old_data = keep_old_data
        self.img_inter = INTERPOLATION_TYPE[img_inter]
        self.mask_inter = INTERPOLATION_TYPE[mask_inter]

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
            self._maps += [(mapx, mapy)]

    def get_data(self):
        """
        Starts augmentation process.
        Returns
        -------
        two arrays
            Augmented images and masks.
        """
        imgs, masks = self._data.get_data()

        new_imgs, new_masks = [], []
        for img, mask in zip(imgs, masks):
            for mapx, mapy in self._maps:
                new_imgs.append(cv2.remap(img, mapx, mapy, self.img_inter, borderMode=cv2.BORDER_REFLECT101))
                new_masks.append(cv2.remap(mask, mapx, mapy, self.mask_inter, borderMode=cv2.BORDER_REFLECT101))

        if self.keep_old_data:
            new_imgs += imgs
            new_masks += masks

        return new_imgs, new_masks

    def __call__(self, data: Augmentor):
        super().__call__(data)
        self._generate_maps()
        return self


class AffineAugment(AugmentOp):
    def __init__(
            self, delta=10., num_matrices=5, seed=None, noise_type='uniform',
            img_inter='linear', mask_inter='nearest',
            keep_old_data=True
    ):
        """
        Performs random affine transformations like rotation, shift, stretching and shrinkage.
        Parameters
        ----------
        delta : float
            Affect how much the final image will be curved.
        num_matrices : int
            For the sake of optimization the number of deformations is fixed.
            There will be `num_matrices`*`img_count` new images.
        seed : int (optional)
            Seed for the random generator that will generate noise tensors for the
            elastic transformation maps.
        noise_type : str
            The noise distribution. Can be 'uniform' or 'gaussian'
        img_inter : str
            Image interpolation type. Can be 'nearest', 'linear' or 'cubic'.
        mask_inter : str
            Image interpolation type. Can be 'nearest', 'linear' or 'cubic'.
        keep_old_data : bool
            Set to false if you don't want to include unaugmented images into the final data set.
        """
        super().__init__()
        self.delta = delta
        self.num_matrices = num_matrices
        self.noise_type = noise_type
        self.random_state = np.random.RandomState(seed)
        self.keep_old_data = keep_old_data
        self.img_inter = INTERPOLATION_TYPE[img_inter]
        self.mask_inter = INTERPOLATION_TYPE[mask_inter]

    def _generate_matrices(self):
        self.mxs = []
        for _ in range(self.num_matrices):
            shape_size = self._img_shape[:2]

            # Random affine
            center_square = np.float32(shape_size) // 2
            square_size = min(shape_size) // 3
            pts1 = np.float32(
                [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                 center_square - square_size])
            noise_shift = self.random_state.uniform(-self.delta, self.delta, size=pts1.shape).astype(np.float32)
            if self.noise_type == 'gaussian':
                noise_shift = self.random_state.randn(*pts1.shape).astype(np.float32) * self.delta
            pts2 = pts1 + noise_shift
            M = cv2.getAffineTransform(pts1, pts2)
            self.mxs.append(M)

    def get_data(self):
        imgs, masks = self._data.get_data()

        shape_size = self._img_shape[:2]
        new_imgs, new_masks = [], []
        for img, mask in zip(imgs, masks):
            for M in self.mxs:
                new_imgs.append(
                    cv2.warpAffine(
                        img, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101,
                        flags=self.img_inter
                    )
                )
                new_masks.append(
                    cv2.warpAffine(
                        mask, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101,
                        flags=self.mask_inter
                    )
                )

        if self.keep_old_data:
            new_imgs += imgs
            new_masks += masks

        return new_imgs, new_masks

    def __call__(self, data: Augmentor):
        super().__call__(data)
        self._generate_matrices()
        return self

