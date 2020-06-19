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

from makiflow.base.maki_entities import MakiTensor
from makiflow.base.maki_entities import MakiCore
from makiflow.generators.simple_generative_model import SGMIterator


class SimpleGenerativeModelBasic(MakiCore, ABC):

    INPUT_MT = 'input_mt'
    OUTPUT_MT = 'output_mt'
    NAME = 'name'
    INPUT_IMAGES = 'input_images'

    def __init__(self, input_x: MakiTensor,
                 output_x: MakiTensor,
                 name="SimpleGenerativeModel"
    ):
        self.name = str(name)
        graph_tensors = output_x.get_previous_tensors()
        graph_tensors.update(output_x.get_self_pair())
        super().__init__(graph_tensors, outputs=[output_x], inputs=[input_x])

        self._training_vars_are_ready = False
        self._sep_loss = None
        self._generator = None

    def predict(self, x):
        return self._session.run(
            self._output_data_tensors[0],
            feed_dict={self._input_data_tensors[0]: x}
        )

    def _get_model_info(self):
        return {
            SimpleGenerativeModelBasic.NAME: self.name,
            SimpleGenerativeModelBasic.INPUT_MT: self._inputs[0].get_name(),
            SimpleGenerativeModelBasic.OUTPUT_MT: self._outputs[0].get_name()
        }

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------SETTING UP TRAINING-------------------------------------

    def _prepare_training_vars(self):
        if not self._set_for_training:
            super()._setup_for_training()
        # VARIABLES PREPARATION
        out_shape = self._outputs[0].get_shape()
        self._out_h = out_shape[1]
        self._out_w = out_shape[2]
        self._batch_sz = out_shape[0]

        self._input_images = self._input_data_tensors[0]
        if self._generator is not None:
            self._target_images = self._generator.get_iterator()[SGMIterator.TARGET_IMAGE]
        else:
            self._target_images = tf.placeholder(tf.float32,
                                                 shape=out_shape,
                                                 name=SimpleGenerativeModelBasic.INPUT_IMAGES
            )

        self._training_out = self._training_outputs[0]

        self._training_vars_are_ready = True

    def set_generator(self, generator):
        self._generator = generator

