from .athena import Athena
from .hermes import Hermes
from .tensorboard import TensorBoard
from .loss_builder import Loss
from .trainer_builder import TrainerBuilder
from abc import ABC


class MakiTrainer(Athena, ABC):
    pass


del Athena
del ABC

