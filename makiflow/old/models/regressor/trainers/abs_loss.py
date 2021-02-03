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
from makiflow.core import TrainerBuilder, LossFabric


class AbsTrainer(RegressorTrainer):
    TYPE = 'AbsTrainer'

    ABS_LOSS = 'ABS_LOSS'

    def _build_local_loss(self, prediction, label):
        abs_loss = LossFabric.abs_loss(label, prediction, raw_tensor=True)
        final_loss = tf.reduce_mean(abs_loss)
        return final_loss


TrainerBuilder.register_trainer(AbsTrainer)
