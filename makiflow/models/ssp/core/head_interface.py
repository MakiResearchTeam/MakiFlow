from abc import ABC, abstractmethod


class HeadInterface(ABC):
    @abstractmethod
    def get_bbox_configuration(self) -> list:
        pass

    @abstractmethod
    def get_coords(self):
        pass

    @abstractmethod
    def get_point_indicators(self):
        pass

    @abstractmethod
    def get_human_indicators(self):
        pass

    @abstractmethod
    def get_grid_size(self) -> list:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass
