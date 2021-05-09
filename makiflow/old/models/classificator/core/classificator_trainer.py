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

from makiflow.core import MakiTrainer
import tensorflow as tf
from abc import ABC


class ClassificatorTrainer(MakiTrainer, ABC):
    WEIGHT_MAP = 'WEIGHT_MAP'
    LABELS = 'LABELS'
    SMOOTHING_LABELS = 'SMOOTHING_LABELS'

    def _init(self):
        super()._init()
        logits_makitensor = super().get_model().get_logits()
        self._logits_name = logits_makitensor.name
        self._num_classes = logits_makitensor.shape()[-1]
        self._labels = super().get_label_tensors()[ClassificatorTrainer.LABELS]
        self._weight_map = super().get_label_tensors()[ClassificatorTrainer.WEIGHT_MAP]

    def get_labels(self):
        return self._labels

    def get_weight_map(self):
        return self._weight_map

    def get_logits(self):
        return super().get_traingraph_tensor(self._logits_name)

    def get_num_classes(self):
        assert self._num_classes is not None
        return self._num_classes

    def _setup_label_placeholders(self):
        logits = super().get_model().get_logits()
        logits_shape = logits.shape()
        return {
            ClassificatorTrainer.LABELS: tf.placeholder(
                dtype=tf.int32,
                shape=[super().get_batch_size(), *logits_shape[1:-1]],
                name=ClassificatorTrainer.LABELS
            ),
            ClassificatorTrainer.WEIGHT_MAP: tf.placeholder(
                dtype=tf.float32,
                shape=[super().get_batch_size(), *logits_shape[1:-1]],
                name=ClassificatorTrainer.WEIGHT_MAP
            )
        }

    def get_label_feed_dict_config(self):
        return {
            self._labels: 0
        }

    def set_params(self, params):
        self.set_smoothing_labels(params.get(ClassificatorTrainer.SMOOTHING_LABELS))

    def set_smoothing_labels(self, smoothing_labels):
        self._smoothing_labels = smoothing_labels

