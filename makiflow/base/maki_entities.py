from abc import abstractmethod
import tensorflow as tf
import json
from copy import copy


class MakiLayer:
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

    @abstractmethod
    def to_dict(self):
        """
        Returns
        -------
        dictionary
            Contains all the necessary information for restoring the layer object.
        """
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
    def __init__(self, data_tensor: tf.Tensor, parent_layer: MakiLayer, parent_tensor_names: list, previous_tensors: dict):
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

    def get_parent_tensors(self)->list:
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

    def get_self_pair(self) -> dict:
        return {self.__name: self}

    def get_name(self):
        return self.__name

    def to_dict(self):
        parent_layer_dict = self.__parent_layer.to_dict()
        return {
            'name': self.__name,
            'parent_tensor_names': self.__parent_tensor_names,
            'parent_layer_info': parent_layer_dict
        }


class MakiModel:
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
        init_op = tf.variables_initializer(self._params)
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
            'model_info': model_info,
            'graph_info': graph_info
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

    def _setup_for_training(self):
        self._set_for_training = True

        self._trainable_vars = []
        self._trainable_layers = []
        for layer_name in self._graph_tensors:
            self._trainable_layers.append(layer_name)
        self._collect_train_params()
        self._build_training_graph()

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

                    X = layer._training_forward(takes[0] if len(takes) == 1 else takes)
                    
                output_tensors[layer.get_name()] = X
                return X
            else:
                return output_tensors[from_.get_name()]

        self._training_outputs = []
        for output in self._outputs:
            self._training_outputs += [create_tensor(output)]
