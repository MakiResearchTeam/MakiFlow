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
from makiflow.models.classificator.utils import error_rate, sparse_cross_entropy
from copy import copy
import json
from makiflow.core import MakiTensor, MakiModel, MakiBuilder
from makiflow.layers import InputLayer
from makiflow.models.classificator.core.classificator_interface import ClassificatorInterface

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
        self._input = self._input.get_data_tensor()
        self._logits = self._output.get_data_tensor()
        self._softmax_out = tf.nn.softmax(self._logits)

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
        Xtest : ndarray
        Ytest : ndarray

        Returns
        -------

        """
        Xtest = Xtest.astype(np.float32)
        n_batches = Xtest.shape[0] // self._batch_sz

        test_cost = 0
        predictions = np.zeros(len(Xtest))
        for k in tqdm(range(n_batches)):
            Xtestbatch = Xtest[k * self._batch_sz:(k + 1) * self._batch_sz]
            Ytestbatch = Ytest[k * self._batch_sz:(k + 1) * self._batch_sz]
            Yish_test_done = self._session.run(self._softmax_out, feed_dict={self._input: Xtestbatch}) + EPSILON
            test_cost += sparse_cross_entropy(Yish_test_done, Ytestbatch)
            predictions[k * self._batch_sz:(k + 1) * self._batch_sz] = np.argmax(Yish_test_done, axis=-1)

        error_r = error_rate(predictions, Ytest)
        test_cost = test_cost / (len(Xtest) // self._batch_sz)
        return error_r, test_cost

    def predict(self, Xtest, use_softmax=True):
        if use_softmax:
            out = self._softmax_out
        else:
            out = self._logits
        n_batches = len(Xtest) // self._batch_sz

        predictions = []
        for i in tqdm(range(n_batches)):
            Xbatch = Xtest[i * self._batch_sz:(i + 1) * self._batch_sz]
            predictions += [self._session.run(out, feed_dict={self._input: Xbatch})]
        if len(predictions) > 1:
            return np.stack(predictions, axis=0)
        else:
            return predictions[0]

