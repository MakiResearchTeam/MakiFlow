from makiflow.core import MakiModel
from abc import abstractmethod, ABC


class ClassificatorInterface(MakiModel, ABC):
    @abstractmethod
    def get_logits(self):
        """
        Used by the trainer.
        Returns
        -------
        MakiTensor
            The logits of the classificator.
        """
        pass
