from __future__ import absolute_import
import numpy as np
import pandas as pd
import cv2
from makiflow.augmentation.segmentation import ElasticAugment, Data
import os

class GD2BBuilder:
    def __init__(self, path_to_hc_list, path_to_balance_config, path_to_mi, resize=None):
        """
        Parameters
        ----------
        path_to_hc_list : str
            Path to the HC list.
        path_to_balance_config : str
            Path to the balance config.
        path_to_mi : str
            Path to csv file contains pairs { path_to_mask : path_to_image }.

        resize : tuple
            Images will be resized accordingly while loading.
        """
        self._hc_list = pd.DataFrame.from_csv(path_to_hc_list)
        self._balance_c = pd.DataFrame.from_csv(path_to_balance_config)
        self._load_masks_images(path_to_mi, resize)
        self._group_images_masks_by_id()
        self._aug = None

    # noinspection PyAttributeOutsideInit
    def _load_masks_images(self, path_to_mi, resize):
        print('Loading masks and images.')
        IMAGE = 'image'
        mi = pd.DataFrame.from_csv(path_to_mi)

        self._masks_images = {}
        for mask_name, row in mi.iterrows():
            image = cv2.imread(row[IMAGE])
            mask = cv2.imread(mask_name)
            if resize is not None:
                image = cv2.resize(image, resize, cv2.INTER_CUBIC)
                mask = cv2.resize(mask, resize, cv2.INTER_NEAREST)

            self._masks_images[mask_name] = (mask, image)
        print('Finished.')

    def _group_images_masks_by_id(self):
        print('Group masks and images by their ids.')
        HCVG = 'hcvg'
        self._groups = {}
        for row_ind, row in self._hc_list.iterrows():
            self._groups[row[HCVG]] = [self._masks_images[row_ind]] + self._groups.get(row[HCVG], [])
        print('Finished.')

    # noinspection PyAttributeOutsideInit
    def set_elastic_aug_params(
            self, img_shape,
            alpha=500, std=8, noise_invert_scale=5,
            img_inter='linear', mask_inter='nearest', border_mode='reflect'
    ):
        self._aug_alpha = alpha
        self._aug_std = std
        self._aug_noise_invert_scale = noise_invert_scale
        self._aug_img_inter = img_inter
        self._aug_mask_inter = mask_inter
        self._aug_border_mode = border_mode
        self._img_shape = img_shape

    def _create_augment(self):
        self._aug = ElasticAugment(
            alpha=self._aug_alpha,
            std=self._aug_std,
            num_maps=1,
            noise_invert_scale=self._aug_noise_invert_scale,
            img_inter=self._aug_img_inter,
            mask_inter=self._aug_mask_inter,
            border_mode=self._aug_border_mode
        )
        self._aug.setup_augmentor(self._img_shape)
        print('Augmentor created.')

    def create_batch(self, path_to_save):
        """
        path_to_save : str
            Path to the folder where the results of the data processing will be saved.
            Example: '.../balanced_batch'.
        """
        for hcv_group in self._groups:
            imgs, masks = self._balance_group(hcv_group)
            self._save_imgs(imgs, masks, hcv_group, path_to_save)
            print(f'{hcv_group} ready')

    def _balance_group(self, hcv_group):
        print(f'Balancing group {hcv_group}...')
        imgs, masks = [], []
        img_ind = 0
        hcvg_cardinality = self._balance_c['0'][hcv_group]
        while hcvg_cardinality > 0:
            im, mask = self._groups[hcv_group][img_ind]
            im, mask = self._augment(im, mask)
            imgs.append(im)
            masks.append(mask)
            hcvg_cardinality -= 1
            img_ind += 1
            if img_ind == len(self._groups[hcv_group]):
                img_ind = 0
                print('Update augmentor...')
                self._create_augment()

        self._aug = None
        print(f'Finished.')
        return imgs, masks

    def _save_imgs(self, imgs, masks, hcv_group, path_to_save):
        masks_path = os.path.join(path_to_save, 'masks')
        imgs_path = os.path.join(path_to_save, 'images')
        os.makedirs(masks_path, exist_ok=True)
        os.makedirs(imgs_path, exist_ok=True)
        for i, (img, mask) in enumerate(zip(imgs, masks)):
            cv2.imwrite(masks_path+f'/{hcv_group}_{i}.bmp', mask)
            cv2.imwrite(imgs_path + f'/{hcv_group}_{i}.bmp', img)

    def _augment(self, im, mask):
        if self._aug is None:
            return im, mask
        data = Data(images=[im], masks=[mask])
        im, mask = self._aug(data).get_data()
        return im[0], mask[0]
