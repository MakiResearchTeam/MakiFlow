from abc import abstractmethod
import tensorflow as tf
from copy import copy
import numpy as np
from makiflow.base.maki_entities import MakiLayer, MakiTensor


class BatchNormBaseLayer(MakiLayer):
    def __init__(self, D, decay, eps, name, use_gamma, use_beta,type_norm, mean, var, gamma, beta):
        """
        Batch Noramlization Procedure:
            X_normed = (X - mean) / variance
            X_final = X*gamma + beta
        gamma and beta are defined by the NN, e.g. they are trainable.

        Parameters
        ----------
        D : int
            Number of tensors to be normalized.
        decay : float
            Decay (momentum) for the moving mean and the moving variance.
        eps : float
            A small float number to avoid dividing by 0.
        use_gamma : bool
            Use gamma in batchnorm or not.
        use_beta : bool
            Use beta in batchnorm or not.
        name : str
            Name of this layer.
        :param mean - batch mean value. Used for initialization mean with pretrained value.
        :param var - batch variance value. Used for initialization variance with pretrained value.
        :param gamma - batchnorm gamma value. Used for initialization gamma with pretrained value.
        :param beta - batchnorm beta value. Used for initialization beta with pretrained value.
        """
        self.D = D
        self.decay = decay
        self.eps = eps
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.running_mean = mean
        self.running_variance = var

        # These variables are needed to change the mean and variance of the batch after
        # the batchnormalization: result*gamma + beta
        # beta - offset
        # gamma - scale
        if beta is None:
            beta = np.zeros(D)
        if gamma is None:
            gamma = np.ones(D)

        params = []
        named_params_dict = {}
        name = str(name)

        # Create gamma
        if use_gamma:
            self.name_gamma = '{}Gamma_{}_id_'.format(type_norm,D) + name
            self.gamma = tf.Variable(gamma.astype(np.float32), name=self.name_gamma)
            named_params_dict[self.name_gamma] = self.gamma
            params += [self.gamma]
        else:
            self.gamma = None

        # Create beta
        if use_beta:
            self.name_beta = '{}Beta_{}_id_'.format(type_norm,D) + name
            self.beta = tf.Variable(beta.astype(np.float32), name=self.name_beta)
            named_params_dict[self.name_beta] = self.beta
            params += [self.beta]
        else:
            self.beta = None

        super().__init__(name, params, named_params_dict)

    def __call__(self, x):
        data = x.get_data_tensor()

        self._init_train_params(data)

        data = self._forward(data)

        parent_tensor_names = [x.get_name()]
        previous_tensors = copy(x.get_previous_tensors())
        previous_tensors.update(x.get_self_pair())
        maki_tensor = MakiTensor(
            data_tensor=data,
            parent_layer=self,
            parent_tensor_names=parent_tensor_names,
            previous_tensors=previous_tensors,
        )
        return maki_tensor

    @abstractmethod
    def _init_train_params(self, data):
        pass

    @abstractmethod
    def _training_forward(self, X):
        pass

    @abstractmethod
    def _forward(self, X):
        pass

    @abstractmethod
    def to_dict(self):
        pass
