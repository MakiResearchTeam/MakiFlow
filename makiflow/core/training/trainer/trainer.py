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

from .train_graph_compiler import GraphCompiler
from .utils import print_train_info, moving_average
from .utils import new_optimizer_used, loss_is_built
from .utils import pack_data, IteratorCloser
from .tensorboard import GradientVariablesWatcher
from ..core import AbstractLoss
from makiflow.core.inference import Model


class Trainer(GraphCompiler):
    # Contains fit loops
    TRAINING_LOSS = 'TRAINING_LOSS'

    def __init__(self, model: Model, train_inputs: list, loss: AbstractLoss):
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
        if loss.loss is not None:
            print('The provided loss object has already been built. It may cause errors when calling fit method '
                  'as the loss computation might be done in another graph requiring data supply for  '
                  'already unreachable tensors.\n'
                  'To avoid that you need to recreate the loss object.')
        self._loss = loss
        self._label_tensors = []
        self._label_tensors += loss.unique_label_tensors.values()
        self._input_tensors = [x.tensor for x in train_inputs]
        super().__init__(model, train_inputs)
        self._track_losses = {}
        self._training_loss = None
        self._tracker = GradientVariablesWatcher(model)
        self._optimizer = None
        self._grads_and_vars = None
        self._losses = []

    @property
    def label_tensors(self):
        return self._label_tensors.copy()

    @property
    def input_tensors(self):
        return self._input_tensors.copy()

    @property
    def training_loss(self):
        assert self._training_loss is not None, 'Training loss is not built.'
        return self._training_loss

    def get_tracker(self):
        return self._tracker

    def compile(self):
        """
        Compiles the training graph and the training loss.
        """
        super().compile_training_graph()
        self.build_loss()

    def add_loss(self, loss: tf.Tensor, label_tensors: list, loss_name: str = None):
        """
        Adds the loss to the losses buffer. The final loss will be computed as a sum of all the losses in the buffer.

        Parameters
        ----------
        loss : tf.Tensor
            The loss scalar.
        label_tensors : list
            List of placeholders which will be used for feed the label data into the training graph.
            If multiple losses reuse the same placeholder, it must be passed in only once.
        loss_name : str
            If provided, the loss will be tracked.
        """
        assert len(loss.shape) == 0, f'The loss must be a scalar, but is a tensor: {label_tensors}'
        self._losses.append(loss)
        self._label_tensors += label_tensors

        if loss_name is not None:
            self.track_loss(loss_tensor=loss, loss_name=loss_name)

    def build_loss(self):
        """
        Builds the training loss and adds it to the track list.
        """
        # noinspection PyAttributeOutsideInit
        loss = None
        if self._loss is not None:
            loss = self._loss.build(self)
        assert loss is not None, 'build method of the Loss instance returned None, but must return the loss scalar.'
        self._training_loss = self._build_final_loss(loss)
        self.track_loss(self._training_loss, Trainer.TRAINING_LOSS)
        loss_is_built()

    def _build_final_loss(self, loss):
        assert loss is not None or len(self._losses) > 0, 'No loss is provided. ' \
                                                          'Please add training loss using add_loss method.'
        if loss is None:
            loss = 0.0

        return loss + tf.reduce_sum(self._losses)

    def track_loss(self, loss_tensor, loss_name, _label_tensors=None):
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
        _label_tensors : list
            List of placeholders which will be used for feed the label data into the training graph.
            If multiple losses reuse the same placeholder, it must be passed in only once.
        """
        loss = self._track_losses.get(loss_name)
        if loss is not None:
            print(f'Overriding already existing {loss_name} loss tensor.')

        self._track_losses[loss_name] = loss_tensor
        self._tracker.add_scalar(loss_tensor, loss_name)

        if _label_tensors is not None:
            print(f'Tensors {_label_tensors} are passed as label_tensors into the track_loss method.')
            print('Make sure the data generator will supply these tensors with data.')
            self._label_tensors += _label_tensors

    def get_track_losses(self):
        return self._track_losses.copy()

    def fit(
            self, optimizer, epochs=1, iter=10, generator=None, print_period=None, global_step=None,
            _progress_bar=True, _print=True
    ):
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
        _progress_bar : bool
            Set to False to turn off tqdm usage within training loop. Default is True.
        _print : bool
            Set to False to turn off loss info printing after each `print_period` iterations.
        Returns
        -------
        dict
            Dictionary with values of the tracked losses.
        """
        if print_period is None:
            print_period = iter

        # Loss value collectors. They will collect all the loss values during this training cycle.
        loss_collectors = {}
        for loss_name in self.get_track_losses():
            loss_collectors[loss_name] = []

        # Loss value holders. They will hold an interpolated loss value for one iteration.
        # This loss value will then be passed to an appropriate loss value collector.
        loss_holders = {}
        for loss_name in self.get_track_losses():
            loss_holders[loss_name] = 0.0

        # This context manager is used to prevent tqdm from breaking in case of exception
        with IteratorCloser() as ic:
            for i in range(epochs):
                if _progress_bar:
                    it = tqdm(range(iter))
                    ic.set_iterator(it)
                else:
                    it = range(iter)

                # Performs training iterations
                for j in it:
                    input_data, labels = None, None
                    # Fetch data from the generator if provided
                    if generator is not None:
                        input_data, labels = next(generator)

                    tracked_losses_vals, summary = self.train_step(
                        optimizer=optimizer, global_step=global_step, input_data=input_data, label_data=labels
                    )
                    # Interpolate loss values and collect them
                    for loss_name in tracked_losses_vals:
                        loss_holders[loss_name] = moving_average(loss_holders[loss_name],
                                                                 tracked_losses_vals[loss_name], j)
                        loss_collectors[loss_name].append(loss_holders[loss_name])

                    self._tracker.increment()
                    if (j + 1) % print_period == 0:
                        name_loss = list(loss_holders.items())
                        if _print:
                            print_train_info(
                                i,
                                *name_loss
                            )
                        self._tracker.write_summary(summary)

        return loss_collectors

    def train_step(self, optimizer, global_step, input_data=None, label_data=None, _feed_dict=None):
        """
        A single training step.

        Parameters
        ----------
        optimizer : tf.train.Optimizer
        global_step
            Please refer to TensorFlow documentation about the global step for more info.
        input_data : tuple or list
            Data for the model's inputs.
        label_data : tuple or list
            Data required for loss (or other) computation.
        _feed_dict : dict
            A dictionary which maps input tensors to the actual data to be fed into the network.
            Examples: { placeholder: np.ndarray }
        Returns
        -------
        dict
            Dictionary containing values of tracked losses.
        tf.Summary
            tf.Summary of the tracked losses.
        """
        # Pack the data into feed_dict
        feed_dict = None
        if input_data is not None and label_data is not None:
            packed_data = pack_data(self.input_tensors, input_data)
            packed_labels = pack_data(self.label_tensors, label_data)
            packed_data.update(packed_labels)
            feed_dict = packed_data

        if _feed_dict is not None:
            feed_dict = _feed_dict

        train_op = self.__minimize_loss(optimizer, global_step)
        track_losses = self.get_track_losses()
        # Summary is created after creation of the train_op
        total_summary = self._tracker.get_total_summary()

        sess = super().get_session()
        tracked_losses_vals, summary, _ = sess.run(
            [track_losses, total_summary, train_op],
            feed_dict=feed_dict
        )
        return tracked_losses_vals, summary

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
            self._tracker.set_vars_grads(vars_and_grads)
            self._tracker.setup_tensorboard()

        self._train_op = optimizer.apply_gradients(
            grads_and_vars=self._grads_and_vars, global_step=global_step
        )

        self.get_session().run(tf.variables_initializer(optimizer.variables()))
        new_optimizer_used()
