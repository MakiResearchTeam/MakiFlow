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
from makiflow.core import MakiTensor, MakiModel


def mf_model2pb(model: MakiModel, path_to_save, model_save_name, output_tensors=None, input_tensors=None):
    """
    Save model graph into pb file
    Show input/output tensors for easier load from pb file

    Parameters
    ----------
    model : MakiModel
        Model from which graph will be saved
    path_to_save : str
        Path to save folder
        Example: '/home/user'
    model_save_name : str
        Path of pb model which will be saved
        Example: 'model'
    output_tensors : list
        Output tensors from model
        By default equal to None, i.e. all output tensors will be taken from model
    input_tensors : list
        Input tensors from model
        By default equal to None, i.e. all input tensors will be taken from model

    """
    print('Output tensors: ')
    tf_output_tensors = []
    if output_tensors is None:
        for elem in model.get_outputs():
            if isinstance(elem, MakiTensor):
                elem = elem.get_data_tensor()

            print('t: ', elem.name)
            tf_output_tensors.append(elem)
    else:
        for elem in output_tensors:
            if isinstance(elem, MakiTensor):
                elem = elem.get_data_tensor()

            print('t: ', elem.name)
            tf_output_tensors.append(elem)
    print('----------------')
    print('Input tensors: ')
    tf_input_tensors = []
    if input_tensors is None:
        for elem in model.get_inputs():
            if isinstance(elem, MakiTensor):
                elem = elem.get_data_tensor()

            print('t: ', elem.name)
            tf_input_tensors.append(elem)
    else:
        for elem in input_tensors:
            if isinstance(elem, MakiTensor):
                elem = elem.get_data_tensor()

            print('t: ', elem.name)
            tf_input_tensors.append(elem)
    print('----------------')

    frozen_graph = model.freeze_graph(output_tensors=tf_output_tensors)
    tf.train.write_graph(frozen_graph, path_to_save, f'{model_save_name}.pb', as_text=False)

    print('Done!')

