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

from makiflow.core import MakiTensor
from makiflow.layers import InputLayer
from makiflow.core.inference import MakiModel
from abc import ABC

EPSILON = np.float32(1e-37)


class CParams:
    INPUT_MT = 'input_mt'
    OUTPUT_MT = 'output_mt'
    NAME = 'name'


class ClassificatorBasis(MakiModel):
    def get_feed_dict_config(self) -> dict:
        return {
            self._input: 0
        }

    def __init__(self, in_x: InputLayer, out_x: MakiTensor, name='MakiClassificator'):
        self._input = in_x
        graph_tensors = copy(out_x.get_previous_tensors())
        # Add output tensor to `graph_tensors` since it doesn't have it.
        # It is assumed that graph_tensors contains ALL THE TENSORS graph consists of.
        graph_tensors.update(out_x.get_self_pair())
        outputs = [out_x]
        inputs = [in_x]
        super().__init__(outputs, inputs)
        self.name = str(name)
        self._batch_sz = in_x.get_shape()[0]
        self._images = self._input_data_tensors[0]
        self._inference_out = self._output_data_tensors[0]
        self._softmax_out = tf.nn.softmax(self._inference_out)
        # For training
        self._training_vars_are_ready = False
        # Identity transformation
        self._labels_transform = lambda x: x
        self._labels = None

    def _get_model_info(self):
        input_mt = self._inputs[0]
        output_mt = self._outputs[0]
        return {
            CParams.INPUT_MT: input_mt.get_name(),
            CParams.OUTPUT_MT: output_mt.get_name(),
            CParams.NAME: self.name
        }

    def evaluate(self, Xtest, Ytest):
        Xtest = Xtest.astype(np.float32)
        Yish_test = tf.nn.softmax(self._inference_out)
        n_batches = Xtest.shape[0] // self._batch_sz

        test_cost = 0
        predictions = np.zeros(len(Xtest))
        for k in tqdm(range(n_batches)):
            Xtestbatch = Xtest[k * self._batch_sz:(k + 1) * self._batch_sz]
            Ytestbatch = Ytest[k * self._batch_sz:(k + 1) * self._batch_sz]
            Yish_test_done = self._session.run(Yish_test, feed_dict={self._images: Xtestbatch}) + EPSILON
            test_cost += sparse_cross_entropy(Yish_test_done, Ytestbatch)
            predictions[k * self._batch_sz:(k + 1) * self._batch_sz] = np.argmax(Yish_test_done, axis=1)

        error_r = error_rate(predictions, Ytest)
        test_cost = test_cost / (len(Xtest) // self._batch_sz)
        return error_r, test_cost

    def predict(self, Xtest, use_softmax=True):
        if use_softmax:
            out = self._softmax_out
        else:
            out = self._inference_out
        n_batches = len(Xtest) // self._batch_sz

        predictions = []
        for i in tqdm(range(n_batches)):
            Xbatch = Xtest[i * self._batch_sz:(i + 1) * self._batch_sz]
            predictions += [self._session.run(out, feed_dict={self._images: Xbatch})]
        if len(predictions) > 1:
            return np.vstack(predictions, axis=0)
        else:
            return predictions[0]

