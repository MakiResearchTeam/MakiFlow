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
from ..core import RegressorTrainer
from makiflow.core import TrainerBuilder, Loss


class AbsTrainer(RegressorTrainer):
    TYPE = 'AbsCETrainer'

    ABS_LOSS = 'ABS_LOSS'

    def _build_loss(self):
        abs_loss = Loss.abs_loss(super().get_labels(), super().get_logits(), raw_tensor=True)

        if self._use_weight_mask:
            weights = super().get_weight_map()
            final_loss = tf.reduce_mean(abs_loss * weights)
        else:
            final_loss = tf.reduce_mean(abs_loss)

        super().track_loss(final_loss, AbsTrainer.ABS_LOSS)
        return final_loss


TrainerBuilder.register_trainer(AbsTrainer)
