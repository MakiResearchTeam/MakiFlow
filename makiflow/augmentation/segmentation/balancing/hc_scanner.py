from __future__ import absolute_import
from makiflow.augmentation.segmentation.balancing.utils import hcv_to_num, get_unique, to_hc_vec
import numpy as np
import cv2
import pandas as pd


# Has-Class Scanner
class HCScanner:
    def __init__(self, masks, num_classes):
        """
        Parameters
        ----------
        masks: dictionary or list
            Dictionary case: contains pairs 'mask name : mask'.
            List case: contains paths to the masks.
        """
        if isinstance(masks, list):
            self.__load_masks(masks)
        else:
            self.masks = masks
        self.num_classes = num_classes

    def __load_masks(self, paths):
        self.masks = {}
        for path in paths:
            self.masks[path] = cv2.imread(path)

    def scan(self):
        # { Mask's name : HC vector group }
        self.masks_hcvg = {}
        # { HC vector group : number of vectors }
        self.hcv_groups = {}
        self.uniq_hcv = []
        for mask_name in self.masks:
            classes = np.unique(self.masks[mask_name])
            hc_vec = to_hc_vec(self.num_classes, classes)
            hc_id = hcv_to_num(hc_vec)

            self.masks_hcvg[mask_name] = hc_id
            self.hcv_groups[hc_id] = 1 + self.hcv_groups.get(hc_id, 0)
            if self.hcv_groups[hc_id] == 1:
                self.uniq_hcv.append(hc_vec)

        self.uniq_hcv = np.vstack(self.uniq_hcv)

    def save_info(self, uniq_hvc_path, masks_hcvg_path):
        pd.DataFrame(self.uniq_hcv).to_csv(uniq_hvc_path)
        pd.DataFrame.from_dict(self.masks_hcvg, orient='index', columns=['hcvg']).to_csv(masks_hcvg_path)
        print('Saved!')
