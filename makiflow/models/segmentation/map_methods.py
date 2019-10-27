import tensorflow as tf
from abc import abstractmethod
from makiflow.models.segmentation.gen_api import MapMethod


class LoadAndResize(MapMethod):
    def __init__(self, image_shape, mask_shape, image_size, mask_size):
        self.image_shape = image_shape
        self.mask_shape = mask_shape
        self.image_size = image_size
        self.mask_size = mask_size

    def load_data(self):
        pass


