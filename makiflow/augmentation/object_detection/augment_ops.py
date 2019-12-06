from __future__ import absolute_import
from makiflow.augmentation.base import AugmentOp, Augmentor
import cv2
import numpy as np


class FlipAugment(AugmentOp):
    FLIP_HORIZONTALLY = 1
    FLIP_VERTICALLY = 0

    def __init__(self, flip_type_list, keep_old_data=True):
        """
        Flips the image and the corresponding bounding boxes.
        Parameters
        ----------
        flip_type_list : list or tuple
            Add to final dataset image with entered type of flip
            Available options:
                FlipAugment.FLIP_HORIZONTALLY;
                FlipAugment.FLIP_VERTICALLY
        keep_old_data : bool
            Set to false if you don't want to include unaugmented images into the final data set.
        """
        super().__init__()
        self.flip_type_list = flip_type_list
        self.keep_old_data = keep_old_data

    def get_data(self):
        """
        Starts augmentation process.
        Returns
        -------
        two arrays
            Augmented images and masks.
        """
        img_w = self._img_shape[1]  # [img_h, img_w, channels]
        old_imgs, old_bboxes, old_classes = self._data.get_data()

        new_imgs, new_bboxes, new_classes = [], [], []
        for img, bboxes, classes in zip(old_imgs, old_bboxes, old_classes):
            for flip_type in self.flip_type_list:
                # Append images
                new_imgs.append(cv2.flip(img, flip_type))
                # Flip bboxes
                new_bboxs = np.copy(bboxes)
                new_bboxs[:, 0] = img_w - bboxes[:, 2]
                new_bboxs[:, 2] = img_w - bboxes[:, 0]
                new_bboxes.append(new_bboxs)
                # Append classes
                new_classes.append(classes)

        if self.keep_old_data:
            new_imgs += old_imgs
            new_bboxes += old_bboxes
            new_classes += old_classes

        return new_imgs, new_bboxes, new_classes

