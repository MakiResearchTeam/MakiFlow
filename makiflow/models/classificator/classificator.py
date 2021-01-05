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

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from makiflow.models.classificator.utils import error_rate
from makiflow.core import MakiTensor, MakiBuilder
from makiflow.layers import InputLayer
from makiflow.models.classificator.core.classificator_interface import ClassificatorInterface
from makiflow.generators import data_iterator
EPSILON = np.float32(1e-37)


class Classificator(ClassificatorInterface):
    INPUT = 'in_x'
    OUTPUT = 'out_x'
    NAME = 'name'

    @staticmethod
    def from_json(path: str, input_tensor: MakiTensor = None):
        """Creates and returns ConvModel from json.json file contains its architecture"""
        model_info, graph_info = super().load_architecture(path)

        output_tensor_name = model_info[Classificator.OUTPUT]
        input_tensor_name = model_info[Classificator.INPUT]
        model_name = model_info[Classificator.NAME]

        inputs_outputs = MakiBuilder.restore_graph([output_tensor_name], graph_info)
        out_x = inputs_outputs[output_tensor_name]
        in_x = inputs_outputs[input_tensor_name]
        print('Model is restored!')
        return Classificator(in_x=in_x, out_x=out_x, name=model_name)

    def __init__(self, in_x: InputLayer, out_x: MakiTensor, name='MakiClassificator'):
        """
        A classifier model.

        Parameters
        ----------
        in_x : MakiTensor
            Input layer.
        out_x : MakiTensor
            Output layer (logits(.
        name : str
            Name of the model.
        """
        self._input = in_x
        self._output = out_x
        super().__init__([out_x], [in_x])
        self.name = str(name)
        self._init_inference()

    def _init_inference(self):
        self._batch_sz = self._input.get_shape()[0]
        self._tf_input = self._input.get_data_tensor()
        self._tf_logits = self._output.get_data_tensor()
        self._softmax_out = tf.nn.softmax(self._tf_logits)

    def get_logits(self):
        return self._output

    def get_feed_dict_config(self) -> dict:
        return {
            self._input: 0
        }

    def _get_model_info(self):
        return {
            Classificator.INPUT: self._input.get_name(),
            Classificator.OUTPUT: self._output.get_name(),
            Classificator.NAME: self.name
        }

    def evaluate(self, Xtest, Ytest):
        """
        Evaluates the model.

        Parameters
        ----------
        Xtest : ndarray of shape [n, ...]
            The input data.
        Ytest : ndarray of shape [n]
            The labels.

        Returns
        -------
        float
            Error rate.
        """
        process = lambda x: np.argmax(x + EPSILON, axis=-1)
        predictions = [process(x) for x in self.predict(Xtest)]
        predictions = np.concatenate(predictions, axis=0)
        error_r = error_rate(predictions, Ytest)
        return error_r

    def predict(self, Xtest, use_softmax=True):
        """
        Performs prediction on the given data.

        Parameters
        ----------
        Xtest : arraylike of shape [n, ...]
            The input data.
        use_softmax : bool
            Whether to use softmax or not.

        Returns
        -------
        arraylike
            Predictions.
        """
        out = self._softmax_out if use_softmax else self._tf_logits
        batch_size = self._batch_sz if self._batch_sz is not None else 1
        predictions = []
        for Xbatch in tqdm(data_iterator(Xtest, batch_size=batch_size)):
            predictions += [self._session.run(out, feed_dict={self._tf_input: Xbatch})]
        return predictions

