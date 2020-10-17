from .athena import Athena
from .hermes import Hermes
from .tensorboard import TensorBoard
from abc import ABC


class MakiTrainer(Athena, ABC):
    pass


del Athena
del ABC

