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

import tensorflow as tf
from makiflow.core import MakiTrainer


class CETrainingModule(MakiTrainer):
    LABELS = 'LABELS'

    def _setup_for_training(self):
        super()._setup_for_training()
        self._labels = super().get_label_tensors()[CETrainingModule.LABELS]
        logits_makitensor = super().get_model().get_logits()
        self._logits_training_tensor = super().get_traingraph_datatensor(logits_makitensor.get_name())

    def _setup_label_placeholders(self):
        return {
            CETrainingModule.LABELS: tf.placeholder(dtype=tf.int32, shape=[super().get_batch_size()])
        }

    def _build_loss(self):
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self._labels,
            logits=self._logits_training_tensor
        )
        return tf.reduce_mean(ce_loss)

    def get_label_feed_dict_config(self):
        return {
            self._labels: 0
        }




