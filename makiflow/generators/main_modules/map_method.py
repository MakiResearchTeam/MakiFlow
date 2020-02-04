from abc import abstractmethod


class MapMethod:
    @abstractmethod
    def load_data(self, data_paths) -> dict:
        pass


class PostMapMethod(MapMethod):
    def __init__(self):
        self._parent_method = None

    @abstractmethod
    def load_data(self, data_paths) -> dict:
        pass

    def __call__(self, parent_method: MapMethod):
        self._parent_method = parent_method
        return self
