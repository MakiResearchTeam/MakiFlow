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
from abc import ABC

from makiflow.core.maki_entities import MakiTensor
from makiflow.core.maki_entities import MakiCore
from makiflow.generators.regressor import RIterator


class RegressorBasic(MakiCore, ABC):

    INPUT_MT = 'input_mt'
    OUTPUT_MT = 'output_mt'
    NAME = 'name'
    INPUT_IMAGES = 'input_images'
    WEIGHT_MASK_IMAGES = 'weight_mask_images'

    def __init__(self, input_x: MakiTensor,
                 output_x: MakiTensor,
                 name="Regressor",
                 use_weight_mask_for_training=False
    ):
        """
        Create Regressor which provides API to train and tests different models.

        Parameters
        ----------
        input_x : MakiTensor
            Input MakiTensor
        output_x : MakiTensor
            Output MakiTensor
        name : str
            Name of this model
        use_weight_mask_for_training : bool
            If set to True, so what weight mask will be used in training
        """
        self.name = str(name)
        graph_tensors = output_x.get_previous_tensors()
        graph_tensors.update(output_x.get_self_pair())
        super().__init__(graph_tensors, outputs=[output_x], inputs=[input_x])

        self._training_vars_are_ready = False
        self._use_weight_mask_for_training = use_weight_mask_for_training
        self._generator = None

    def predict(self, x):
        """
        Get result from neural network according to certain input

        Parameters
        ----------
        x: ndarray
            Input for neural network, i. e. for this model.

        Returns
        ----------
        ndarray
            Output of the neural network
        """
        return self._session.run(
            self._output_data_tensors[0],
            feed_dict={self._input_data_tensors[0]: x}
        )

    def _get_model_info(self):
        return {
            RegressorBasic.NAME: self.name,
            RegressorBasic.INPUT_MT: self._inputs[0].get_name(),
            RegressorBasic.OUTPUT_MT: self._outputs[0].get_name()
        }

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------SETTING UP TRAINING-------------------------------------

    def _prepare_training_vars(self):
        if not self._set_for_training:
            super()._setup_for_training()
        # VARIABLES PREPARATION
        out_shape = self._outputs[0].get_shape()
        self._batch_sz = out_shape[0]

        self._input_x = self._input_data_tensors[0]
        if self._generator is not None:
            self._target_x = self._generator.get_iterator()[RIterator.TARGET_X]
        else:
            self._target_x = tf.placeholder(tf.float32,
                                            shape=out_shape,
                                            name=RegressorBasic.INPUT_IMAGES
            )

        # Weight mask
        if self._use_weight_mask_for_training:
            if self._generator is not None:
                self._weight_mask = self._generator.get_iterator()[RIterator.WEIGHTS_MASK]
            else:
                self._weight_mask = tf.placeholder(tf.float32,
                                                   shape=out_shape,
                                                   name=RegressorBasic.WEIGHT_MASK_IMAGES
                )
        else:
            self._weight_mask = None

        self._training_out = self._training_outputs[0]

        self._training_vars_are_ready = True

    def set_generator(self, generator):
        """
        Set generator (i. e. pipeline) for this model

        Parameters
        ----------
        generator : mf.generators
            Certain generator for this model
        """
        self._generator = generator

