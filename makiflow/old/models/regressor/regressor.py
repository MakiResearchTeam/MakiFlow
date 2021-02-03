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

import numpy as np
from tqdm import tqdm
from makiflow.core import MakiTensor, MakiBuilder, MakiModel
from makiflow.core.training.trainer.utils import pack_data
from .core import RegressorInterface
from makiflow.generators import data_iterator

EPSILON = np.float32(1e-37)


class Regressor(RegressorInterface):
    INPUTS = 'in_x'
    OUTPUTS = 'out_x'
    NAME = 'name'

    @staticmethod
    def from_json(path: str, input_tensor: MakiTensor = None):
        """Creates and returns ConvModel from json.json file contains its architecture"""
        model_info, graph_info = MakiModel.load_architecture(path)

        output_names = model_info[Regressor.OUTPUTS]
        input_names = model_info[Regressor.INPUTS]
        model_name = model_info[Regressor.NAME]

        inputs_outputs = MakiBuilder.restore_graph(output_names, graph_info)
        out_x = [inputs_outputs[name] for name in output_names]
        in_x = [inputs_outputs[name] for name in input_names]
        print('Model is restored!')
        return Regressor(in_x=in_x, out_x=out_x, name=model_name)

    def __init__(self, in_x: list, out_x: list, name='MakiClassificator'):
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
        super().__init__(out_x, in_x)
        self.name = str(name)
        self._batch_sz = super().get_inputs()[0].get_shape()[0]

    def get_logits(self):
        return super().get_outputs()

    def get_feed_dict_config(self) -> dict:
        feed_dict_config = {}
        for i, x in enumerate(super().get_inputs()):
            feed_dict_config[x] = i

        return feed_dict_config

    def _get_model_info(self):
        return {
            Regressor.INPUTS: [in_x.get_name() for in_x in super().get_inputs()],
            Regressor.OUTPUTS: [out_x.get_name() for out_x in super().get_outputs()],
            Regressor.NAME: self.name
        }

    def predict(self, *args):
        """
        Performs prediction on the given data.

        Parameters
        ----------
        Xtest : arraylike of shape [n, ...]
            The input data.

        Returns
        -------
        arraylike
            Predictions.

        """
        feed_dict_config = self.get_feed_dict_config()
        batch_size = self._batch_sz if self._batch_sz is not None else 1
        predictions = []
        for data in tqdm(data_iterator(*args, batch_size=batch_size)):
            packed_data = pack_data(feed_dict_config, data)
            predictions += [
                self._session.run(
                    [out_x.get_data_tensor() for out_x in super().get_outputs()],
                    feed_dict=packed_data)
            ]
        new_pred = []
        for i in range(len(super().get_outputs())):
            single_preds = []
            for output in predictions:
                # grab i-th data
                single_preds.append(output[i])
            new_pred.append(np.concatenate(single_preds, axis=0)[:len(args[0])])

        return new_pred
