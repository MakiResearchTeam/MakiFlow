from ..main_modules import SSPInterface
from makiflow.generators.pipeline.gen_base import GenLayer
from makiflow.generators.ssp import SSPIterator


class VanillaTrainer:
    def __init__(self, model: SSPInterface, generator: GenLayer):
        model.training_on()
