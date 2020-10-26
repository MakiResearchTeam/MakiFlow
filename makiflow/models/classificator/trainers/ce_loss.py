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
from ..core import ClassificatorTrainer
from makiflow.core import TrainerBuilder


class CETrainer(ClassificatorTrainer):
    TYPE = 'CETrainer'
    CROSS_ENTROPY = 'CROSS_ENTROPY'

    def _build_loss(self):
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=super().get_labels(),
            logits=super().get_logits()
        )
        ce_loss = tf.reduce_mean(ce_loss)
        super().track_loss(ce_loss, CETrainer.CROSS_ENTROPY)
        return ce_loss


TrainerBuilder.register_trainer(CETrainer)
