import tensorflow as tf
from abc import abstractmethod


class MapMethod:
    @abstractmethod
    def load_data(self, paths):
        pass