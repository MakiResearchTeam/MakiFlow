import json
from abc import abstractmethod, ABC


class MakiRestorable(ABC):
    NAME = 'NAME'
    TYPE = 'TYPE'
    PARAMS = 'PARAMS'

    @abstractmethod
    def to_dict(self):
        pass

    @staticmethod
    @abstractmethod
    def build(**kwargs):
        pass


