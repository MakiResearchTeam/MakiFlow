from __future__ import absolute_import
from makiflow.augmentation.segmentation.base import Augmentor


class Data(Augmentor):
    def __init__(self, images, masks):
        super().__init__()
        self.images = images
        self.masks = masks
        self._img_shape = images[0].shape

    def get_data(self):
        return self.images, self.masks


