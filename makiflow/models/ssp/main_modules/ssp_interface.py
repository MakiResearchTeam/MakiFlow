from abc import ABC, abstractmethod


class SSPInterface(ABC):
    @abstractmethod
    def get_heads(self):
        pass

    @abstractmethod
    def training_on(self):
        pass
