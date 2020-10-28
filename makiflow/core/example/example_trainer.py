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

from __future__ import absolute_import
from makiflow.core.training import MakiTrainer, Loss
import tensorflow as tf


class ExampleTrainer(MakiTrainer):
    LABELS = 'LABELS'

    def _setup_label_placeholders(self):
        return {
            ExampleTrainer.LABELS: tf.placeholder(dtype='float32', shape=[None], name='labels')
        }

    def get_label_feed_dict_config(self):
        return {
            self._labels: 0
        }

    def _init_(self):
        # Always call the super()._setup_for_training() first
        super()._init_()
        # Define here all the necessary variables
        self._labels = super().get_label_tensors()[ExampleTrainer.LABELS]

    def _build_loss(self):
        # This method must return a scalar of the training loss
        out_xs = self.get_model().get_outputs()
        losses = []
        for out_x in out_xs:
            out_t = out_x.get_data_tensor()
            losses += [Loss.mse_loss(self._labels, out_t)]

        return tf.add_n(losses)
