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

from .graph_entities import MakiTensor


def to_makitensor(tf_tensor, name, parent_layer=None, parent_tensor_names=None, previous_tensors=None):
    """
    Converts a TensorFlow tensor to MakiTensor.
    Parameters
    ----------
    tf_tensor : tf.Tensor
        Actual data tensor.
    name : str
        Name of the tensor.
    parent_layer : MakiLayer
        Layer that produced this tensor. If it is set to None, it may cause an error during architecture
        save of the model.
    parent_tensor_names : list
        List of names of the MakiTensors that were used during creation of this tensor.
    previous_tensors : dict
        Dictionary containing all the tensors appearing earlier in the computational graph.

    Returns
    -------
    MakiTensor
    """
    if parent_tensor_names is None:
        parent_tensor_names = []
    if previous_tensors is None:
        previous_tensors = {}

    return MakiTensor(
        data_tensor=tf_tensor,
        parent_layer=parent_layer,
        name=name,
        parent_tensor_names=parent_tensor_names,
        previous_tensors=previous_tensors
    )


