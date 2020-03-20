from .maki_builder import MakiBuilder
from .maki_trainer import MakiTrainer
from abc import ABC


class MakiCore(MakiBuilder, MakiTrainer, ABC):
    pass
