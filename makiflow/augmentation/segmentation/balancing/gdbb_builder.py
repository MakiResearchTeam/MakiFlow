from __future__ import absolute_import
import numpy as np
import pandas as pd
import cv2
from makiflow.augmentation.segmentation import ElasticAugment, Data


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
        self.hc_list = pd.DataFrame.from_csv(path_to_hc_list)
        self.balance_c = pd.DataFrame.from_csv(path_to_balance_config)
        self._load_masks_images(path_to_mi, resize)
        self._group_images_masks_by_id()
        self.aug = None

    # noinspection PyAttributeOutsideInit
    def _load_masks_images(self, path_to_mi, resize):
        IMAGE = 'image'
        mi = pd.DataFrame.from_csv(path_to_mi)

        self.masks_images = {}
        for mask_name, row in mi.iterrows():
            image = cv2.imread(row[IMAGE])
            mask = cv2.imread(mask_name)
            if resize is not None:
                image = cv2.resize(image, resize)
                mask = cv2.resize(mask, resize)

            self.masks_images[mask_name] = (mask, image)

    def _group_images_masks_by_id(self):
        HCVG = 'hcvg'
        self.groups = {}
        for row_ind, row in self.hc_list.iterrows():
            self.groups[row[HCVG]] = [self.masks_images[row_ind]] + self.groups.get(row[HCVG], [])

    # noinspection PyAttributeOutsideInit
    def set_elastic_aug_params(
            self, alpha=500, std=8, noise_invert_scale=5,
            img_inter='linear', mask_inter='nearest', border_mode='reflect'
    ):
        self.aug_alpha = alpha
        self.aug_std = std
        self.aug_noise_invert_scale = noise_invert_scale
        self.aug_img_inter = img_inter
        self.aug_mask_inter = mask_inter
        self.aug_border_mode = border_mode

    def _create_augment(self):
        self.aug = ElasticAugment(
            alpha=self.aug_alpha,
            std=self.aug_std,
            num_maps=1,
            noise_invert_scale=self.aug_noise_invert_scale,
            img_inter=self.aug_img_inter,
            mask_inter=self.aug_mask_inter,
            border_mode=self.aug_border_mode
        )

    def create_batch(self, path_to_save):
        """
        path_to_save : str
            Path to the folder where the results of the data processing will be saved.
        """
        for hcv_group in self.groups:
            imgs, masks = self._balance_group(hcv_group)
            self._save_imgs(imgs, masks, hcv_group, path_to_save)
            print(f'{hcv_group} ready')

    def _balance_group(self, hcv_group):
        imgs, masks = [], []
        img_ind = 0
        hcvg_cardinality = self.balance_c[hcv_group]
        while hcvg_cardinality > 0:
            im, mask = self.groups[hcv_group][img_ind]
            im, mask = self._augment(im, mask)
            imgs.append(im)
            masks.append(mask)
            hcvg_cardinality -= 1
            img_ind += 1
            if img_ind == len(self.groups[hcv_group]):
                img_ind = 0
                self._create_augment()

        self.aug = None
        return imgs, masks

    def _save_imgs(self, imgs, masks, hcv_group, path_to_save):
        for i, img, mask in enumerate(zip(imgs, masks)):
            cv2.imwrite(path_to_save+f'/_{hcv_group}_{i}_mask.bmp', mask)
            cv2.imwrite(path_to_save + f'/_{hcv_group}_{i}_img.bmp', img)

    def _augment(self, im, mask):
        if self.aug is None:
            return im, mask
        data = Data(images=[im], masks=[mask])
        im, mask = self.aug(data).get_data()
        return im[0], mask[0]
