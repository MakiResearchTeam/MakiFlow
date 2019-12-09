from __future__ import absolute_import
from makiflow.augmentation.base import AugmentOp
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


class ContrastBrightnessAugment(AugmentOp):
    clip_int = 'int'
    clip_float = 'float'

    def __init__(self, params, clip_type='int', keep_old_data=True):
        """
        Adjusts brightness and contrast according to `params`
        Parameters
        ----------
        params : list
            List of tuples (alpha, beta). The pixel values will be changed using the following formula:
            new_pix_value = old_pix_value * alpha + beta.
        clip_type : str
            'int' - the images will be clipped within [0, 255] range.
            'float' - the images will be clipped within [0, 1] range.
        """
        super().__init__()
        self.params = params
        self.clip_type = clip_type
        self.keep_old_data = keep_old_data

    def get_data(self):
        old_imgs, old_bboxes, old_classes = self._data.get_data()
        new_imgs = []
        new_bboxes = old_bboxes
        new_classes = old_classes
        for alpha, beta in self.params:
            for image in old_imgs:
                new_image = image * alpha + beta
                if self.clip_type == ContrastBrightnessAugment.clip_int:
                    new_image = np.clip(new_image, 0, 255).astype(np.uint8)
                elif self.clip_type == ContrastBrightnessAugment.clip_float:
                    new_image = np.clip(new_image, 0.0, 1.0).astype(np.float32)
                else:
                    raise RuntimeError(f'Uknown clip type: {self.clip_type}')
                new_imgs += [new_image]

        if self.keep_old_data:
            new_imgs += old_imgs
            new_bboxes += old_bboxes
            new_classes += old_classes
        return new_imgs, new_bboxes, new_classes
