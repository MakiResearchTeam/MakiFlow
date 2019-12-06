from __future__ import absolute_import
from makiflow.augmentation.base import Augmentor


class Data(Augmentor):
    def __init__(self, images, bboxes, classes):
        """
        Parameters
        ----------
        images : list
            List of images (ndarrays).
        bboxes : list
            List of ndarrays (concatenated bboxes [x1, y1, x2, y2]).
        classes : list
            List of ndarrays (arrays of the classes of the `bboxes`).
        """
        super().__init__()
        self.images = images
        self.bboxes = bboxes
        self.classes = classes
        self._img_shape = images[0].shape

    def get_data(self):
        return self.images, self.bboxes, self.classes

