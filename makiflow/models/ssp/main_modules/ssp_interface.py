from abc import abstractmethod
from makiflow.core.inference import MakiCore


class SSPInterface(MakiCore):
    @abstractmethod
    def get_heads(self):
        pass
