from abc import abstractmethod, ABC
import tensorflow as tf
import json
from copy import copy

from makiflow.layers.trainable_layers import TrainableLayerAddress
from makiflow.layers.untrainable_layers import UnTrainableLayerAddress, InputLayer
from makiflow.layers.rnn_layers import RNNLayerAddress


class MakiRestorable(ABC):
    TYPE = 'Restorable'
    PARAMS = 'params'
    FIELD_TYPE = 'type'
    NAME = 'name'

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
    def __init__(self, name, params, named_params_dict):
        self._name = name
        self._params = params
        self._named_params_dict = named_params_dict

    @abstractmethod
    def __call__(self, x):
        """
        Parameters
        ----------
        x: MakiTensor or list of MakiTensors

        Returns
        -------
        MakiTensor or list of MakiTensors
        """
        pass

    @abstractmethod
    def _training_forward(self, x):
        pass

    def get_params(self):
        return self._params

    def get_params_dict(self):
        """
        This data is used for correct saving and loading models using TensorFlow checkpoint files.
        """
        return self._named_params_dict

    def get_name(self):
        return self._name


class MakiTensor:
    NAME = 'name'
    PARENT_TENSOR_NAMES = 'parent_tensor_names'
    PARENT_LAYER_INFO = 'parent_layer_info'

    def __init__(self, data_tensor: tf.Tensor, parent_layer: MakiLayer, parent_tensor_names: list,
                 previous_tensors: dict):
        self.__data_tensor: tf.Tensor = data_tensor
        self.__name: str = parent_layer.get_name()
        self.__parent_tensor_names = parent_tensor_names
        self.__parent_layer = parent_layer
        self.__previous_tensors: dict = previous_tensors

    def get_data_tensor(self):
        return self.__data_tensor

    def get_parent_layer(self):
        """
        Returns
        -------
        Layer
            Layer which produced current MakiTensor.
        """
        return self.__parent_layer

    def get_parent_tensors(self) -> list:
        """
        Returns
        -------
        list of MakiTensors
            MakiTensors that were used for creating current MakiTensor.
        """
        parent_tensors = []
        for name in self.__parent_tensor_names:
            parent_tensors += [self.__previous_tensors[name]]
        return parent_tensors

    def get_parent_tensor_names(self):
        return self.__parent_tensor_names

    def get_previous_tensors(self) -> dict:
        """
        Returns
        -------
        dict of MakiTensors
            All the MakiTensors that appear earlier in the computational graph.
            The dictionary contains pairs: { name of the tensor: MakiTensor }.
        """
        return self.__previous_tensors

    def get_shape(self):
        return self.__data_tensor.get_shape().as_list()

    def get_self_pair(self) -> dict:
        return {self.__name: self}

    def __str__(self):
        name = self.__name
        shape = self.get_shape()
        dtype = self.__data_tensor._dtype.name

        return f"MakiTensor(name={name}, shape={shape}, dtype={dtype})"

    def __repr__(self):
        name = self.__name
        shape = self.get_shape()
        dtype = self.__data_tensor._dtype.name

        return f"<mf.base.MakiTensor 'name={name}' shape={shape} dtype={dtype}>"

    def get_name(self):
        return self.__name

    def to_dict(self):
        parent_layer_dict = self.__parent_layer.to_dict()
        return {
            MakiTensor.NAME: self.__name,
            MakiTensor.PARENT_TENSOR_NAMES: self.__parent_tensor_names,
            MakiTensor.PARENT_LAYER_INFO: parent_layer_dict
        }


class MakiModelRestoreBase:
    # -----------------------------------------------------------LAYERS RESTORATION-----------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __layer_from_dict(layer_dict):
        """
        Creates and returns Layer from dictionary
        """

        # Collects the address to all existing layers
        all_layers_adress = {}
        all_layers_adress.update(RNNLayerAddress.ADDRESS_TO_CLASSES)
        all_layers_adress.update(TrainableLayerAddress.ADDRESS_TO_CLASSES)
        all_layers_adress.update(UnTrainableLayerAddress.ADDRESS_TO_CLASSES)

        params = layer_dict[MakiRestorable.PARAMS]

        build_layer = all_layers_adress.get(layer_dict[MakiRestorable.FIELD_TYPE])

        if build_layer is None:
            raise KeyError(f'{layer_dict[MakiRestorable.FIELD_TYPE]} was not found!')

        return build_layer.build(params)

    # -----------------------------------------------------------GRAPH RESTORATION--------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def restore_graph(outputs, graph_info_json, batch_sz=None, generator=None):
        """
        Rectore Inference graph with inputs and outputs of model from json.
        """
        # dict {NameTensor : Info about this tensor}
        graph_info = {}

        for tensor in graph_info_json:
            graph_info[tensor[MakiRestorable.NAME]] = tensor

        used = {}
        coll_tensors = {}

        def restore_in_and_out_x(from_):
            """
            Rectore inputs and outputs of model from json.
            """
            # from_ - name of layer
            parent_layer_info = graph_info[from_]
            if used.get(from_) is None:
                used[from_] = True
                # like "to"
                all_parent_names = parent_layer_info[MakiTensor.PARENT_TENSOR_NAMES]
                # store ready tensors
                takes = []
                if len(all_parent_names) != 0:
                    # All layer except input layer
                    layer = Builder.__layer_from_dict(parent_layer_info[MakiTensor.PARENT_LAYER_INFO])
                    for elem in all_parent_names:
                        takes += [restore_in_and_out_x(elem)]
                    answer = layer(takes[0] if len(takes) == 1 else takes)
                else:
                    # Input layer
                    temp = {}
                    temp.update({
                        MakiRestorable.FIELD_TYPE: parent_layer_info[MakiRestorable.FIELD_TYPE],
                        MakiRestorable.PARAMS: parent_layer_info[MakiRestorable.PARAMS]}
                    )
                    if batch_sz is not None:
                        temp[MakiRestorable.PARAMS][InputLayer.INPUT_SHAPE][0] = batch_sz
                    if generator is not None:
                        answer = generator
                    else:
                        answer = MakiModelRestoreBase.__layer_from_dict(temp)

                coll_tensors[from_] = answer
                return answer
            else:
                return coll_tensors[from_]

        for name_output in outputs:
            restore_in_and_out_x(name_output)

        return coll_tensors

    @staticmethod
    @abstractmethod
    def from_json(path_to_model):
        """
        Rectore certain model.
        This method must be implemented by other models.
        """
        pass


class MakiModel(MakiModelRestoreBase):
    MODEL_INFO = 'model_info'
    GRAPH_INFO = 'graph_info'

    def __init__(self, graph_tensors: dict, outputs: list, inputs: list):
        self._graph_tensors = graph_tensors
        self._outputs = outputs
        self._inputs = inputs
        self._set_for_training = False

        self._output_data_tensors = []
        for maki_tensor in self._outputs:
            self._output_data_tensors += [maki_tensor.get_data_tensor()]

        self._input_data_tensors = []
        for maki_tensor in self._inputs:
            self._input_data_tensors += [maki_tensor.get_data_tensor()]

        self._collect_params()

    def _collect_params(self):
        self._params = []
        self._named_dict_params = {}
        for tensor_name in self._graph_tensors:
            layer = self._graph_tensors[tensor_name].get_parent_layer()
            self._params += layer.get_params()
            self._named_dict_params.update(layer.get_params_dict())

    def set_session(self, session: tf.Session):
        self._session = session
        params = []
        # Do not initialize variables using self._params since
        # self._params contains only gradient descent trainable parameters.
        # It is not the case with BatchNormalization where mean and variance are
        # computed on the fly.
        for key in self._named_dict_params:
            params += [self._named_dict_params[key]]
        init_op = tf.variables_initializer(params)
        self._session.run(init_op)

    def load_weights(self, path, layer_names=None):
        """
        This function uses default TensorFlow's way for restoring models - checkpoint files.
        Example: '/home/student401/my_model/model.ckpt'
        Parameters
        ----------
            path : str
                Full path to stored weights
            layer_names : list of str
                Names of layer which weights need load from file into model
        """
        vars_to_load = {}
        if layer_names is not None:
            for layer_name in layer_names:
                layer = self._graph_tensors[layer_name].get_parent_layer()
                vars_to_load.update(layer.get_params_dict())
        else:
            vars_to_load = self._named_dict_params
        saver = tf.train.Saver(vars_to_load)
        saver.restore(self._session, path)
        print('Weights are loaded.')

    def save_weights(self, path, layer_names=None):
        """
        This function uses default TensorFlow's way for saving models - checkpoint files.
        Example: '/home/student401/my_model/model.ckpt'
        Parameters
        ----------
            path : str
                Full path to place where weights should saved
            layer_names : list of str
                Names of layer which weights need save from model
        """
        vars_to_save = {}
        if layer_names is not None:
            for layer_name in layer_names:
                layer = self._graph_tensors[layer_name].get_parent_layer()
                vars_to_save.update(layer.get_params_dict())
        else:
            vars_to_save = self._named_dict_params
        saver = tf.train.Saver(vars_to_save)
        save_path = saver.save(self._session, path)
        print(f'Weights are saved to {save_path}')

    def save_architecture(self, path):
        """
        This function save architecture of model in certain path.
        Example: '/home/student401/my_model/architecture.json'
        Parameters
        ----------
            path : str
                Full path to place where architecture should saved
        """
        model_info = self._get_model_info()
        graph_info = self._get_graph_info()
        model_dict = {
            MakiModel.MODEL_INFO: model_info,
            MakiModel.GRAPH_INFO: graph_info
        }

        model_json = json.dumps(model_dict, indent=1)
        json_file = open(path, mode='w')
        json_file.write(model_json)
        json_file.close()
        print(f"Model's architecture is saved to {path}.")

    @abstractmethod
    def _get_model_info(self):
        """
        This method must be implemented by other models.
        """
        pass

    def _get_graph_info(self):
        tensor_dicts = []
        for tensor_name in self._graph_tensors:
            tensor = self._graph_tensors[tensor_name]
            tensor_dicts += [tensor.to_dict()]
        return tensor_dicts

    def get_node(self, node_name):
        return self._graph_tensors.get(node_name)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------MAKIMODEL TRAINING------------------------------

    def set_layers_trainable(self, layers):
        """
        Parameters
        ----------
        layers: list
            List of tuples: (layer_name, bool).
            Example: to make layer called `conv1` untrainable use
            the following tuple (`conv1`, False), if you want to make it trainable
            use the following tuple (`conv1`, True).
        """
        if not self._set_for_training:
            self._setup_for_training()
        for layer_name, is_trainable in layers:
            if is_trainable and layer_name not in self._trainable_layers:
                self._trainable_layers.append(layer_name)
            elif not is_trainable and layer_name in self._trainable_layers:
                self._trainable_layers.remove(layer_name)

        self._collect_train_params()

    def _collect_train_params(self):
        self._trainable_vars.clear()
        for layer_name in self._trainable_layers:
            layer = self._graph_tensors[layer_name].get_parent_layer()
            self._trainable_vars += layer.get_params()
        # Create graph or refresh it
        self._build_training_graph()

    def _setup_for_training(self):
        self._set_for_training = True
        
        # Collect all the layers names since all of them are trainable from
        # the beginning.
        self._trainable_layers = []
        for layer_name in self._graph_tensors:
            self._trainable_layers.append(layer_name)
        self._trainable_vars = []
        self._collect_train_params()
        
        # Setup L2 regularization
        self._uses_l2_regularization = False
        self._l2_reg_loss_is_build = False
        self._l2_regularized_layers = {}
        for layer_name in self._trainable_layers:
            self._l2_regularized_layers[layer_name] = 1e-6  # This value seems to be proper as a default

        self._uses_l1_regularization = False
        self._l1_reg_loss_is_build = False
        self._l1_regularized_layers = {}
        for layer_name in self._trainable_layers:
            self._l1_regularized_layers[layer_name] = 1e-6  # This value seems to be proper as a default

    # L2 REGULARIZATION

    def set_l2_reg(self, layers):
        """
        Enables L2 regularization while training and allows to set different
        decays to different weights.
        WARNING! It is assummed that `set_layers_trainable` method won't
        be used anymore.

        Parameters
        ----------
        layers : list
            Contains tuples (layer_name, decay) where decay is float number(set it to
            None if you want to turn off regularization on this weight).
        """
        if not self._set_for_training:
            self._setup_for_training()

        self._uses_l2_regularization = True
        
        for layer_name, decay in layers:
            self._l2_regularized_layers[layer_name] = decay
    
    def set_common_l2_weight_decay(self, decay=1e-6):
        """
        Enables L2 regularization while training. 
        `decay` will be set as decay for each regularized weight.
        If you haven't used `set_l2_reg` method and did not turn off
        the regularization on certain layers, the regularization will be
        set on all the trainable layers.

        Parameters
        ----------
        decay : float
            Decay for all the regularized weights.
        """
        if not self._set_for_training:
            self._setup_for_training()

        self._uses_l2_regularization = True

        for layer_name in self._l2_regularized_layers:
            if self._l2_regularized_layers[layer_name] is not None:
                self._l2_regularized_layers[layer_name] = decay
    
    def _build_l2_loss(self):
        self._l2_reg_loss = tf.constant(0.0)
        for layer_name in self._l2_regularized_layers:
            decay = self._l2_regularized_layers[layer_name]
            if decay is not None:
                layer = self._graph_tensors[layer_name].get_parent_layer()
                params = layer.get_params()
                for param in params:
                    self._l2_reg_loss += tf.nn.l2_loss(param) * tf.constant(decay)
        
        self._l2_reg_loss_is_build = True

    # L1 REGULARIZATION

    def set_l1_reg(self, layers):
        """
        Enables L2 regularization while training and allows to set different
        decays to different weights.
        WARNING! It is assummed that `set_layers_trainable` method won't
        be used anymore.

        Parameters
        ----------
        layers : list
            Contains tuples (layer_name, decay) where decay is float number(set it to
            None if you want to turn off regularization on this weight).
        """
        if not self._set_for_training:
            self._setup_for_training()

        self._uses_l1_regularization = True
        
        for layer_name, decay in layers:
            self._l1_regularized_layers[layer_name] = decay
    
    def set_common_l1_weight_decay(self, decay=1e-6):
        """
        Enables L2 regularization while training. 
        `decay` will be set as decay for each regularized weight.
        If you haven't used `set_l1_reg` method and did not turn off
        the regularization on certain layers, the regularization will be
        set on all the trainable layers.

        Parameters
        ----------
        decay : float
            Decay for all the regularized weights.
        """
        if not self._set_for_training:
            self._setup_for_training()

        self._uses_l1_regularization = True

        for layer_name in self._l1_regularized_layers:
            if self._l1_regularized_layers[layer_name] is not None:
                self._l1_regularized_layers[layer_name] = decay
    
    def _build_l1_loss(self):
        self._l1_reg_loss = tf.constant(0.0)
        for layer_name in self._l1_regularized_layers:
            decay = self._l1_regularized_layers[layer_name]
            if decay is not None:
                layer = self._graph_tensors[layer_name].get_parent_layer()
                params = layer.get_params()
                for param in params:
                    self._l1_reg_loss += tf.abs(tf.reduce_sum(param)) * tf.constant(decay)

        self._l1_reg_loss_is_build = True

    def _build_final_loss(self, custom_loss):
        if self._uses_l1_regularization:
            if not self._l1_reg_loss_is_build:
                self._build_l1_loss()
            custom_loss += self._l1_reg_loss

        if self._uses_l2_regularization:
            if not self._l2_reg_loss_is_build:
                self._build_l2_loss()
            custom_loss += self._l2_reg_loss
        
        return custom_loss

    def _build_training_graph(self):
        # Contains pairs {layer_name: tensor}, where `tensor` is output
        # tensor of layer called `layer_name`
        output_tensors = {}
        used = {}

        def create_tensor(from_):
            if used.get(from_.get_name()) is None:
                layer = from_.get_parent_layer()
                used[layer.get_name()] = True
                X = copy(from_.get_data_tensor())
                takes = []
                # Check if we at the beginning of the computational graph, i.e. InputLayer
                if from_.get_parent_tensor_names() is not None:
                    for elem in from_.get_parent_tensors():
                        takes += [create_tensor(elem)]

                    if layer.get_name() in self._trainable_layers:
                        X = layer._training_forward(takes[0] if len(takes) == 1 else takes)
                    else:
                        X = layer._forward(takes[0] if len(takes) == 1 else takes)

                output_tensors[layer.get_name()] = X
                return X
            else:
                return output_tensors[from_.get_name()]

        self._training_outputs = []
        for output in self._outputs:
            self._training_outputs += [create_tensor(output)]
