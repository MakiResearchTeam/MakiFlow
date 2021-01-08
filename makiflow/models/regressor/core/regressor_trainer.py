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


class RegressorTrainer(MakiTrainer, ABC):
    WEIGHT_MAP = 'WEIGHT_MAP'
    LABELS = 'LABELS'

    def _init(self):
        super()._init()
        self._use_weight_mask = False
        logits_makitensor = super().get_model().get_logits()
        self._logits_name = logits_makitensor.get_name()
        self._labels = super().get_label_tensors()[RegressorTrainer.LABELS]
        self._weight_map = super().get_label_tensors()[RegressorTrainer.WEIGHT_MAP]

    def use_weight_mask(self):
        """
        If called, weight mask will be used during loss computation.
        """
        self._use_weight_mask = True

    def get_labels(self):
        return self._labels

    def get_weight_map(self):
        return self._weight_map

    def get_logits(self):
        return super().get_traingraph_tensor(self._logits_name)

    def _setup_label_placeholders(self):
        logits = super().get_model().get_logits()
        logits_shape = logits.get_shape()
        return {
            RegressorTrainer.LABELS: tf.placeholder(
                dtype=tf.float32,
                shape=[super().get_batch_size(), *logits_shape[1:]],
                name=RegressorTrainer.LABELS
            ),
            RegressorTrainer.WEIGHT_MAP: tf.placeholder(
                dtype=tf.float32,
                shape=[super().get_batch_size(), *logits_shape[1:]],
                name=RegressorTrainer.WEIGHT_MAP
            )
        }

    def get_label_feed_dict_config(self):
        final_dict = {
            self._labels: 0
        }
        if self._use_weight_mask:
            final_dict[self._weight_map] = 1

        return final_dict

