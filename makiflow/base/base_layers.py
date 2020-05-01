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

from abc import abstractmethod
import tensorflow as tf
from copy import copy
import numpy as np
from makiflow.base.maki_entities import MakiLayer, MakiTensor


class BatchNormBaseLayer(MakiLayer):
    TYPE = 'BatchNormBaseLayer'
    D = 'D'
    DECAY = 'decay'
    EPS = 'eps'
    USE_BETA = 'use_beta'
    USE_GAMMA = 'use_gamma'
    TRACK_RUNNING_STATS = 'track_running_stats'

    def __init__(self, D, decay, eps, name, use_gamma, use_beta, regularize_gamma, regularize_beta,
                 type_norm, mean, var, gamma, beta, track_running_stats):
        """
        Batch Normalization Procedure:
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
        mean : float
            Batch mean value. Used for initialization mean with pretrained value.
        var : float
            Batch variance value. Used for initialization variance with pretrained value.
        gamma : float
            Batchnorm gamma value. Used for initialization gamma with pretrained value.
        beta : float
            Batchnorm beta value. Used for initialization beta with pretrained value.
        """
        self.D = D
        self.decay = decay
        self.eps = eps
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.running_mean = mean
        self.running_variance = var
        self._track_running_stats = track_running_stats

        self._is_running_vars_created = False

        # These variables are needed to change the mean and variance of the batch after
        # the batchNormaization: result*gamma + beta
        # beta - offset
        # gamma - scale
        if beta is None:
            beta = np.zeros(D)
        if gamma is None:
            gamma = np.ones(D)

        params = []
        regularize_params = []
        named_params_dict = {}
        name = str(name)

        # Create gamma
        if use_gamma:
            self.name_gamma = '{}Gamma_{}_id_'.format(type_norm, D) + name
            self.gamma = tf.Variable(gamma.astype(np.float32), name=self.name_gamma)
            named_params_dict[self.name_gamma] = self.gamma
            params += [self.gamma]
            if regularize_gamma:
                regularize_params += [self.gamma]
        else:
            self.gamma = None

        # Create beta
        if use_beta:
            self.name_beta = '{}Beta_{}_id_'.format(type_norm, D) + name
            self.beta = tf.Variable(beta.astype(np.float32), name=self.name_beta)
            named_params_dict[self.name_beta] = self.beta
            params += [self.beta]
            if regularize_beta:
                regularize_params += [self.beta]
        else:
            self.beta = None
        super().__init__(name, params=params,
                         regularize_params=regularize_params,
                         named_params_dict=named_params_dict
        )

    def __call__(self, x):
        data = x.get_data_tensor()

        if self._track_running_stats and not self._is_running_vars_created:
            self._is_running_vars_created = True
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
