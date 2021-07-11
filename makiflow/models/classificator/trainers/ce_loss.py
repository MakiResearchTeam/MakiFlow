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


if __name__ == '__main__':
    from makiflow.models.classificator import Classificator
    from makiflow.layers import InputLayer
    # SEGMENTATION CASE
    print('SEGMENTATION CASE------------------------------------------------------------------------------------------')
    x = InputLayer(input_shape=[32, 128, 128, 3], name='input')

    model = Classificator(in_x=x, out_x=x)
    trainer = CETrainer(model=model, train_inputs=[x])

    print('LABELS TENSORS:', trainer.get_label_tensors())
    trainer.compile()
    print('LOSS TENSORS:', trainer.get_track_losses())

    # VANILLA CLASSIFICATION CASE
    print('VANILLA CLASSIFICATION CASE--------------------------------------------------------------------------------')
    x = InputLayer(input_shape=[32, 3], name='input')
    model = Classificator(in_x=x, out_x=x)
    trainer = CETrainer(model=model, train_inputs=[x])

    print('LABELS TENSORS:', trainer.get_label_tensors())
    trainer.compile()
    print('LOSS TENSORS:', trainer.get_track_losses())
