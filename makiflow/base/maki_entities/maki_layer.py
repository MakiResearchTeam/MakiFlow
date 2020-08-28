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

from abc import abstractmethod, ABC
from .maki_tensor import MakiTensor


class MakiRestorable(ABC):
    TYPE = 'Restorable'
    PARAMS = 'params'
    FIELD_TYPE = 'type'
    NAME = 'name'

    TRAINING_MODE = 'TrainingGraph'
    INFERENCE_MODE = 'InferenceGraph'

    @staticmethod
    def build(params: dict):
        """
        Parameters
        ----------
        params : dict
            Dictionary of specific params to build layers.

        Returns
        -------
        MakiLayer
            Specific built layers
        """
        pass

    @abstractmethod
    def to_dict(self):
        """
        Returns
        -------
        dictionary
            Contains all the necessary information for restoring the layer object.
        """
        pass


class MakiLayer(MakiRestorable):
    __EXC_OUTPUT_NAMES_NOT_ALIGNED = 'Number of the output tensors and the given outputs names is not aligned.' + \
                                     'Given names {0}, got output tensors {1}.'
    __EXC_OUTPUT_NONE = 'Output of the layer is None. Check whether the `_forward` method works correctly.'

    def __init__(self, name: str, params: list, regularize_params: list, named_params_dict: dict,
                 outputs_names: list = None):
        """
        Base class for all the layers in MakiFlow.
        Parameters
        ----------
        name : str
            Name of the layer.
        params : list
            List of TensorFlow variables - trainable parameters.
        regularize_params : dict
            Dictionary of pairs { var_name : tf.Variable }. All the variables from this dictionary will
            be regularized unless otherwise is specified.
        named_params_dict : dict
            Dictionary of pairs { var_name : tf.Variable }. Must contain all the model's parameters since
            this information is used for model saving.
        outputs_names : list (optional)
            If the layer outputs several tensors, name for those must be provided in this list, otherwise an exception
            will be thrown.
            If the layer outputs a single tensor, then it will inherit its parent layer's name.
            If the layer outputs several tensors, then names in the list will be concatenated with the parent layer's
            name.
        """
        self._name = name
        self._params = params
        self._regularize_params = regularize_params
        self._named_params_dict = named_params_dict
        if outputs_names is None:
            outputs_names = []
        self._outputs_names = outputs_names

    def __check_output(self, output):
        if output is None:
            raise Exception(MakiLayer.__EXC_OUTPUT_NONE)

        if output is list and len(output) != len(self._outputs_names):
            message = MakiLayer.__EXC_OUTPUT_NAMES_NOT_ALIGNED.format(
                len(output), len(self._outputs_names)
            )

            message = message + '\nOutput tensors are:\n'
            for t in output:
                message += f'{t}\n'

            message = message + 'Output names are:\n'
            for name in self._outputs_names:
                message = message + name + '\n'
            raise Exception(message)

    def __call__(self, x):
        """
        Unpacks datatensor(s) (tf.Tensor) from the given MakiTensor(s) `x`, performs layer's transformation and
        wraps out the output of that transformation into a new MakiTensor.
        This implementation is generic and can be used by layers with any transformations:
        1 to 1 (one tensor is transformed into one tensor);
        n to 1 (several tensors are transformed into one tensor);
        n to n (several tensors are transformed into several tensors (rnn)).
        The order of the data tensors is kept during `_forward` call and during returning the output tensors, i.e.
        datatensors have the same ordering as input MakiTensors and output MakiTensors have the same ordering as
        output datatensors.

        Parameters
        ----------
        x: MakiTensor or list of MakiTensors
            The order of the datatensors follow the order of the MakiTensors.
        Returns
        -------
        MakiTensor or list of MakiTensors
        """
        if x is not list:
            x = [x]

        data_tensors = []
        previous_tensors = {}
        parent_tensor_names = []
        for _x in x:
            data_tensors += [_x.get_data_tensor()]
            previous_tensors.update(_x.get_previous_tensors())
            previous_tensors.update(_x.get_self_pair())
            parent_tensor_names += [_x.get_name()]

        if len(data_tensors) == 1:
            data_tensors = data_tensors[0]
        output = self._forward(data_tensors)

        self.__check_output(output)

        # Output MakiTensors
        if output is list:
            output_mt = []
            for i, t, name in enumerate(zip(output, self._outputs_names)):
                output_mt += [
                    MakiTensor(
                        data_tensor=t,
                        parent_layer=self,
                        parent_tensor_names=parent_tensor_names,
                        previous_tensors=previous_tensors,
                        name=self.get_name() + '/' + name,
                        index=i
                    )
                ]
            return output_mt
        else:
            return MakiTensor(
                data_tensor=output,
                parent_layer=self,
                parent_tensor_names=parent_tensor_names,
                previous_tensors=previous_tensors,
                name=self.get_name()
            )

    @abstractmethod
    def _forward(self, x, computation_mode=MakiRestorable.INFERENCE_MODE):
        """
        Method that contains the logic of the transformation that the layer performs.

        Parameters
        ----------
        x : tf.Tensor or list of tf.Tensors
            Actual data tensor(s) to be transformed.
        computation_mode : str
            Used for scoping the operations within the method.

        Returns
        -------
        tf.Tensor or list of tf.Tensors
        """
        pass

    @abstractmethod
    def _training_forward(self, x):
        """
        Used in during the construction of the training graph.
        Logic of some of the layers may change depending on whether the model is
        being trained or used for inference (for example, batchnormalization).

        Parameters
        ----------
        x : tf.Tensor or list of tf.Tensors
            Actual data to be transformed.

        Returns
        -------
        tf.Tensor or list of tf.Tensors
        """
        pass

    def get_params(self):
        """
        Returns
        -------
        list
            Trainable parameters of this layer.
        """
        return self._params

    def get_params_dict(self):
        """
        This data is used for correct saving and loading models using TensorFlow checkpoint files.

        Returns
        -------
        dict
            Dictionary that store name of tensor and tensor itself of this layer.
        """
        return self._named_params_dict

    def get_params_regularize(self):
        """
        This data is used for collect params for regularisation.
        Some of the parameters, like bias, are preferred not to be regularized since it can cause underfitting.
        Thus, it makes sense to track parameters that are being regularized and that are not.

        Returns
        ------
        list
            List of parameters to be regularized.
        """
        return self._regularize_params

    def get_name(self):
        """
        Returns
        -------
        str
            Name of the layer.
        """
        return self._name
