from abc import abstractmethod
from makiflow.base.maki_entities import MakiCore


class SSPInterface(MakiCore):
    @abstractmethod
    def get_heads(self):
        pass
