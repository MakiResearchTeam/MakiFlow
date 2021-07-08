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
from warnings import warn
import tensorflow as tf

from .maki_tensor import MakiTensor
from ..debug.exception_scope import method_exception_scope


class MakiRestorable(ABC):
    TYPE = 'type'
    PARAMS = 'params'
    # old code, refactor some day
    FIELD_TYPE = 'type'
    NAME = 'name'

    TRAINING_MODE = 'TrainingGraph'
    INFERENCE_MODE = 'InferenceGraph'

    # Gurren Lagann
    NO_PARAMETER = 'What the hell do you think this is!?'

    @staticmethod
    def get_param(params: dict, param_name, default_value):
        value = params.get(param_name, MakiRestorable.NO_PARAMETER)
        if value == MakiRestorable.NO_PARAMETER:
            warn(f"Parameter {param_name} wasn't found. The default value is used: default={default_value}")
            return default_value
        return value

    @staticmethod
    @abstractmethod
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
    TRAINING_MODE = 'TrainingGraph'
    INFERENCE_MODE = 'InferenceGraph'

    __EXC_OUTPUT_NAMES_NOT_ALIGNED = 'Number of the output tensors and given outputs names is not aligned.' + \
                                     'Given output names {0}, got output tensors {1}. Make sure you provided names for the output' + \
                                     ' tensors in the MakiLayer constructor (see the `outputs_names` parameter).'
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
        # Counts the number of time the `__call__` method was called.
        self._name = name
        self._params = params
        self._regularize_params = regularize_params
        self._named_params_dict = named_params_dict
        if outputs_names is None:
            outputs_names = []
        self._outputs_names = outputs_names
        self._n_calls = 0
        self._n_calls_training = 0
        # This is used during training graph construction and is a solution for cases
        # when the same layer is being used several times. Unfortunately
        # there is no better solution yet.
        # Dictionary of pairs { parent MakiTensor name : list child MakiTensor name }
        self._children_dict = {}

    @method_exception_scope()
    def __call__(self, x, is_training=False):
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
        if not isinstance(x, list):
            x = [x]

        # --- Make sure all the input data are MakiTensors
        for x_ in x:
            assert isinstance(x_, MakiTensor), f'Expected type MakiTensor, but received {type(x_)}.'

        # --- Gather graph information for the future MakiTensors
        data_tensors = []
        previous_tensors = {}
        parent_tensor_names = []
        for _x in x:
            data_tensors += [_x.tensor]
            previous_tensors.update(_x.previous_tensors)
            previous_tensors.update(_x.get_self_pair())
            parent_tensor_names += [_x.name]

        if len(data_tensors) == 1:
            data_tensors = data_tensors[0]

        # --- Do a forward pass
        if is_training:
            output = self.training_forward(data_tensors)
        else:
            output = self.forward(data_tensors)

        self.__check_output(output)
        # --- Generate output MakiTensors and return them
        if not isinstance(output, tuple):
            output = (output,)
            output_names = [self.name]
        else:
            output_names = [self.name + '/' + name for name in self._outputs_names]

        output_mt = []
        for i, (t, makitensor_name) in enumerate(zip(output, output_names)):
            makitensor_name = self._output_tensor_name(makitensor_name, is_training=is_training)
            output_mt += [
                MakiTensor(
                    data_tensor=t,
                    parent_layer=self,
                    parent_tensor_names=parent_tensor_names,
                    previous_tensors=previous_tensors,
                    name=makitensor_name,
                    index=i
                )
            ]
        if is_training:
            self._n_calls_training += 1
        else:
            self._n_calls += 1
        self._update_children(parent_tensor_names, output_mt)

        if len(output_mt) == 1:
            output_mt = output_mt[0]

        return output_mt

    def __check_output(self, output):
        if output is None:
            raise Exception(MakiLayer.__EXC_OUTPUT_NONE)

        if isinstance(output, tuple) and len(output) != len(self._outputs_names):
            message = MakiLayer.__EXC_OUTPUT_NAMES_NOT_ALIGNED.format(
                len(self._outputs_names), len(output)
            )

            message = message + '\nOutput tensors are:\n'
            for t in output:
                message += f'{t}\n'

            message = message + '\nOutput names are:\n'
            for name in self._outputs_names:
                message = message + name + '\n'
            raise Exception(message)

    def _output_tensor_name(self, name: str, is_training: bool):
        if is_training:
            if self._n_calls_training != 0:
                name += f'_call{self._n_calls_training}'
            name += '_training'
            return name

        if self._n_calls != 0:
            name += f'_call{self._n_calls}'

        return name

    def _update_children(self, parent_tensor_names: list, output_mt):
        if not isinstance(output_mt, list):
            output_mt = [output_mt]

        output_mt_names = []
        for output_tensor in output_mt:
            output_mt_names += [output_tensor.name]

        for parent_tensor_name in parent_tensor_names:
            self._children_dict[parent_tensor_name] = output_mt_names

    @abstractmethod
    def forward(self, x, computation_mode=INFERENCE_MODE):
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
    def training_forward(self, x):
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

    @property
    def name(self):
        """
        Returns
        -------
        str
            Name of the layer.
        """
        return self._name

    def get_n_calls(self):
        """
        Returns
        -------
        int
            The number of times the layer was called.
        """
        return self._n_calls

    def get_children(self, makitensor_name):
        """

        Parameters
        ----------
        makitensor_name : str
            Name of the MakiTensor that was once passed through this layer.

        Returns
        -------
        list
            List of the names of the MakiTensors that were created after passing in a MakiTensor
            with `makitensor_name` name.
        """
        return self._children_dict[makitensor_name]

