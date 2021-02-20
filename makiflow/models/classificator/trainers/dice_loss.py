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
from makiflow.core import TrainerBuilder, Loss


class DiceTrainer(ClassificatorTrainer):
    TYPE = 'DiceTrainer'
    DICE_LOSS = 'DICE_LOSS'

    AXES = 'axes'
    EPS = 'eps'

    def to_dict(self):
        return {
            TrainerBuilder.TYPE: DiceTrainer.TYPE,
            TrainerBuilder.PARAMS: {
                DiceTrainer.AXES: self._axes,
                DiceTrainer.EPS: self._eps
            }
        }

    def set_params(self, params):
        self.set_eps(params[DiceTrainer.EPS])
        self.set_axes(params[DiceTrainer.AXES])

    def _init(self):
        super()._init()
        self._eps = 1e-3
        self._axes = None

    def set_eps(self, eps):
        """
        Used to prevent division by zero in the Dice denominator.

        Parameters
        ----------
        eps : float
            Larger the values
        """
        assert eps >= 0, f'Eps must be non-negative. Received eps={eps}'
        # noinspection PyAttributeOutsideInit
        self._eps = eps

    def set_axes(self, axes):
        """
        Sets which axes the dice value will be computed on. The computed dice values will be averaged
        along the remaining axes.

        Parameters
        ----------
        axes : list
        """
        self._axes = axes

    def _build_loss(self):
        labels = super().get_labels()
        logits = super().get_logits()

        # p - predicted probability
        # g - ground truth label
        p = tf.nn.sigmoid(logits)
        g = labels
        dice_loss = Loss.dice_loss(
            p=p,
            g=g,
            eps=self._eps,
            axes=self._axes
        )

        super().track_loss(dice_loss, DiceTrainer.DICE_LOSS)
        return dice_loss


TrainerBuilder.register_trainer(DiceTrainer)


if __name__ == '__main__':
    from makiflow.models.classificator import Classificator
    from makiflow.layers import InputLayer
    # SEGMENTATION CASE
    print('SEGMENTATION CASE------------------------------------------------------------------------------------------')
    x = InputLayer(input_shape=[32, 128, 128, 3], name='input')

    model = Classificator(in_x=x, out_x=x)
    trainer = DiceTrainer(model=model, train_inputs=[x])

    print('LABELS TENSORS:', trainer.get_label_tensors())
    trainer.compile()
    print('LOSS TENSORS:', trainer.get_track_losses())

    # VANILLA CLASSIFICATION CASE
    print('VANILLA CLASSIFICATION CASE--------------------------------------------------------------------------------')
    x = InputLayer(input_shape=[32, 3], name='input')
    model = Classificator(in_x=x, out_x=x)
    trainer = DiceTrainer(model=model, train_inputs=[x])

    print('LABELS TENSORS:', trainer.get_label_tensors())
    trainer.compile()
    print('LOSS TENSORS:', trainer.get_track_losses())
