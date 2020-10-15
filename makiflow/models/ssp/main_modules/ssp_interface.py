from abc import abstractmethod
from makiflow.core.maki_entities import MakiCore


class SSPInterface(MakiCore):
    @abstractmethod
    def get_heads(self):
        pass
