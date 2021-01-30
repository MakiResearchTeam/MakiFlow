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
from tqdm import tqdm
from abc import abstractmethod

from .utils import print_train_info, moving_average
from .utils import new_optimizer_used, loss_is_built
from .core import TrainingCore
from .gradient_variables_watcher import GradientVariablesWatcher
from makiflow.core.training.utils import pack_data, IteratorCloser
from ..inference import MakiModel
from .loss.core import LossInterface


class ModelFitter(TrainingCore):
    # Contains fit loops
    TRAINING_LOSS = 'TRAINING_LOSS'

    def __init__(self, model: MakiModel, train_inputs: list, loss: LossInterface):
        """
        Provides basic tools for the training setup. Builds final loss tensor and the training graph.

        Parameters
        ----------
        model : MakiModel
            The model's object.
        train_inputs : list
            List of the input training MakiTensors. Their names must be the same as their inference counterparts!
        """
        # Can be required during _setup_for_training call. Thus, create this variable
        # first and then call super init.
        self._loss = loss
        self._label_tensors = loss.get_label_tensors()
        super().__init__(model, train_inputs)
        self._track_losses = {}
        self._training_loss = None
        self._hermes = GradientVariablesWatcher(model)
        self._optimizer = None
        self._grads_and_vars = None

    def get_label_tensors(self):
        """
        Returns
        -------
        dict
            Contains pairs (tensor_name, tf.Tensor) of required tensors of labels.
        """
        return self._label_tensors.copy()

    def get_hermes(self):
        return self._hermes

    def compile(self):
        """
        Compiles the training graph and the training loss.
        """
        super().compile_training_graph()
        self.build_loss()

    def build_loss(self):
        """
        Builds the training loss and adds it to the track list.
        """
        # noinspection PyAttributeOutsideInit
        loss = self._loss.build(self)
        assert loss is not None, '_build_loss method returned None, but must return the loss scalar.'
        self._training_loss = super()._build_final_loss(loss)
        self.track_loss(self._training_loss, ModelFitter.TRAINING_LOSS)
        loss_is_built()

    def track_loss(self, loss_tensor, loss_name):
        """
        Adds loss to the track list. The loss value will be printed in the fit cycle and its value will also
        be shown on the tensorboard.
        Tip: this method can be used to add any scalar value calculated by the tensorflow means.

        Parameters
        ----------
        loss_tensor : tf.Tensor
            Scalar of the loss to track.
        loss_name : str
            Name of the loss.
        """
        loss = self._track_losses.get(loss_name)
        if loss is not None:
            print(f'Overriding already existing {loss_name} loss tensor.')

        self._track_losses[loss_name] = loss_tensor
        self._hermes.add_scalar(loss_tensor, loss_name)

    def get_track_losses(self):
        return self._track_losses.copy()

    def fit(self, optimizer, epochs=1, iter=10, print_period=None, global_step=None):
        """
        Performs fitting of the model.

        Parameters
        ----------
        optimizer : TensorFlow optimizer
            Model uses TensorFlow optimizers in order train itself.
        epochs : int
            Number of epochs to run.
        iter : int
            Number of training iterations per update.
        print_period : int
            Every `print_period` training iterations the training info will be displayed.
        global_step
            Please refer to TensorFlow documentation about the global step for more info.
        Returns
        -------
        dict
            Dictionary with values of the tracked losses.
        """
        train_op = self.__minimize_loss(optimizer, global_step)

        if print_period is None:
            print_period = iter

        # Loss value collectors. They will collect all the loss values during this training cycle.
        loss_collectors = {}
        for loss_name in self.get_track_losses():
            loss_collectors[loss_name] = []

        sess = super().get_session()
        track_losses = self.get_track_losses()
        total_summary = self._hermes.get_total_summary()

        # This context manager is used to prevent tqdm from breaking in case of exception
        with IteratorCloser() as ic:
            for i in range(epochs):
                it = tqdm(range(iter))
                ic.set_iterator(it)
                # Loss value holders. They will hold an interpolated loss value for one iteration.
                # This loss value will then be passed to an appropriate loss value collector.
                loss_holders = {}
                for loss_name in self.get_track_losses():
                    loss_holders[loss_name] = 0.0

                # Performs training iterations
                for j in it:
                    tracked_losses_vals, summary, _ = sess.run(
                        [track_losses, total_summary, train_op]
                    )
                    # Interpolate loss values and collect them
                    for loss_name in tracked_losses_vals:
                        loss_holders[loss_name] = moving_average(loss_holders[loss_name], tracked_losses_vals[loss_name], j)
                        loss_collectors[loss_name].append(loss_holders[loss_name])

                    self._hermes.increment()
                    if (j + 1) % print_period == 0:
                        name_loss = list(loss_holders.items())
                        print_train_info(
                            i,
                            *name_loss
                        )
                        self._hermes.write_summary(summary)

        return loss_collectors

    def fit_generator(self, generator, optimizer, epochs=1, iter=10, print_period=None, global_step=None):
        """
        Performs fitting of the model.

        Parameters
        ----------
        generator : python iterator
            Returns tuple of (data, labels). Data and labels can be tuples or lists themselves.
        optimizer : TensorFlow optimizer
            Model uses TensorFlow optimizers in order train itself.
        epochs : int
            Number of epochs to run.
        iter : int
            Number of training iterations per update.
        print_period : int
            Every `print_period` training iterations the training info will be displayed.
        global_step
            Please refer to TensorFlow documentation about the global step for more info.
        Returns
        -------
        dict
            Dictionary with values of the tracked losses.
        """
        train_op = self.__minimize_loss(optimizer, global_step)

        if print_period is None:
            print_period = iter

        # Loss value collectors. They will collect all the loss values during this training cycle.
        loss_collectors = {}
        for loss_name in self.get_track_losses():
            loss_collectors[loss_name] = []

        sess = super().get_session()
        track_losses = self.get_track_losses()
        total_summary = self._hermes.get_total_summary()
        input_feed_dict = self.get_input_feed_dict_config()
        label_feed_dict = self.get_label_feed_dict_config()

        # This context manager is used to prevent tqdm from breaking in case of exception
        with IteratorCloser() as ic:
            for i in range(epochs):
                it = tqdm(range(iter))
                ic.set_iterator(it)

                # Loss value holders. They will hold an interpolated loss value for one iteration.
                # This loss value will then be passed to an appropriate loss value collector.
                loss_holders = {}
                for loss_name in self.get_track_losses():
                    loss_holders[loss_name] = 0.0

                # Performs training iterations
                for j in it:
                    input_data, labels = next(generator)
                    packed_data = pack_data(input_feed_dict, input_data)
                    packed_labels = pack_data(label_feed_dict, labels)
                    packed_data.update(packed_labels)
                    tracked_losses_vals, summary, _ = sess.run(
                        [track_losses, total_summary, train_op],
                        feed_dict=packed_data
                    )
                    # Interpolate loss values and collect them
                    for loss_name in tracked_losses_vals:
                        loss_holders[loss_name] = moving_average(loss_holders[loss_name], tracked_losses_vals[loss_name], j)
                        loss_collectors[loss_name].append(loss_holders[loss_name])

                    self._hermes.increment()
                    if (j + 1) % print_period == 0:
                        name_loss = list(loss_holders.items())
                        print_train_info(
                            i,
                            *name_loss
                        )
                        self._hermes.write_summary(summary)

        return loss_collectors

    def __minimize_loss(self, optimizer, global_step):
        assert optimizer is not None, 'No optimizer is provided.'
        assert super().is_compiled(), 'The model is not compiled.'

        if self._optimizer != optimizer:
            self.__create_train_op(optimizer, global_step)

        return self._train_op

    def __create_train_op(self, optimizer, global_step):
        self._optimizer = optimizer

        if self._grads_and_vars is None:
            training_vars = super().get_trainable_params()
            # Returns list of tuples: [ (grad, var) ]
            self._grads_and_vars = optimizer.compute_gradients(self._training_loss, training_vars)
            vars_and_grads = [(var, grad) for grad, var in self._grads_and_vars]
            self._hermes.set_vars_grads(vars_and_grads)
            self._hermes.setup_tensorboard()

        self._train_op = optimizer.apply_gradients(
            grads_and_vars=self._grads_and_vars, global_step=global_step
        )

        self.get_session().run(tf.variables_initializer(optimizer.variables()))
        new_optimizer_used()

    def get_input_feed_dict_config(self):
        """
        Returns
        -------
        dict
            The same as the one the model returns via its `get_feed_dict_config` method, except
            that the input tensors are replaced with their counterparts from the training graph.
        """
        model = super().get_model()
        # Feed dict with inference input tensors
        feed_dict_config = model.get_feed_dict_config()
        train_feed_dict_config = dict()
        for t, i in feed_dict_config.items():
            name = t.get_name()
            tensor = super().get_traingraph_tensor(name)
            train_feed_dict_config[tensor] = i
        return train_feed_dict_config

    def get_label_feed_dict_config(self):
        labels = self.get_label_tensors()
        label_feed_dict_config = {}
        for i, t in enumerate(labels.values()):
            label_feed_dict_config[t] = i
        return label_feed_dict_config
