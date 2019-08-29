from __future__ import absolute_import
from makiflow.augmentation.segmentation.base import Augmentor


class Data(Augmentor):
    def __init__(self, images, masks):
        super().__init__(self)
        self.images = images
        self.masks = masks

    def get_data(self):
        return self.images, self.masks


