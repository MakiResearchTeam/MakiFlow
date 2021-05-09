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
from abc import ABC, abstractmethod


class RegressorTrainer(MakiTrainer, ABC):
    WEIGHT_MAP = 'WEIGHT_MAP'
    LABELS = 'LABELS'

    def _init(self):
        super()._init()
        self._use_weight_mask = False
        logits_makitensor = super().get_model().get_logits()
        self._logits_names = [l.name() for l in logits_makitensor]
        self._labels = super().get_label_tensors()

    # noinspection PyAttributeOutsideInit
    def set_loss_sources(self, source_names):
        self._logits_names = source_names

    def get_labels(self):
        return self._labels

    def get_logits(self):
        logits = []
        for name in self._logits_names:
            logits.append(super().get_traingraph_tensor(name))
        return logits

    @abstractmethod
    def _build_local_loss(self, prediction, label):
        pass

    def _build_loss(self):
        losses = []
        for name in self._logits_names:
            prediction = super().get_traingraph_tensor(name)
            label = self.get_labels()[name]
            losses.append(self._build_local_loss(prediction, label))
            super().track_loss(losses[-1], name)
        return tf.add_n([0.0, *losses], name='total_loss')

    def _setup_label_placeholders(self):
        logits = super().get_model().get_logits()
        batch_size = super().get_batch_size()
        label_tensors = {}
        for l in logits:
            label_tensors[l.name()] = tf.placeholder(
                dtype='float32', shape=[batch_size, *l.shape()[1:]], name=f'label_{l.name()}'
            )
        return label_tensors

    def get_label_feed_dict_config(self):
        labels = super().get_label_tensors()
        label_feed_dict_config = {}
        for i, t in enumerate(labels.values()):
            label_feed_dict_config[t] = i
        return label_feed_dict_config

