# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

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
