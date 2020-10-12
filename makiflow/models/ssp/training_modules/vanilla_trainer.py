from ..main_modules import SSPInterface
from makiflow.generators.pipeline.gen_base import GenLayer
from makiflow.generators.ssp import SSPIterator


class VanillaTrainer:
    def __init__(self, model: SSPInterface, generator: GenLayer):
        model.training_on()
        self._model = model
        self._generator = generator
        self._iterator = generator.get_iterator()
        # Training data tensors
        self._training_classification_labels = self._iterator[SSPIterator.CLASS_LABELS]
        self._training_human_presence_labels = self._iterator[SSPIterator.HUMANP_LABELS]
        self._training_points_coords_labels = self._iterator[SSPIterator.POINTS_LABELS]

        self._build_training_tensors()

    def _build_training_tensors(self):
        pass
