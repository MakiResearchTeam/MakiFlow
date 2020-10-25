import tensorflow as tf
from .utils import print_train_info, moving_average
from .utils import new_optimizer_used, loss_is_built
from .core import TrainingCore
from tqdm import tqdm
from abc import abstractmethod
from .hermes import Hermes
from ..utils import pack_data
from ..inference import MakiModel


class Athena(TrainingCore):
    # Athena is the goddess of wisdom and intelligence.
    # This entity is responsible for training the model.
    TRAINING_LOSS = 'TRAINING_LOSS'

    def __init__(self, model: MakiModel, train_inputs: list, label_tensors: dict = None):
        """
        Provides basic tools for the training setup. Builds final loss tensor and the training graph.
        Parameters
        ----------
        model : MakiModel
            The model's object.
        train_inputs : list
            List of the input training MakiTensors. Their names must be the same as their inference counterparts!
        label_tensors : dict
            Contains pairs (tensor_name, tf.Tensor), where tf.Tensor contains the required training data.
        """
        # Can be required during _setup_for_training call. Thus, create this variable
        # first and then call super init.
        self._label_tensors = label_tensors
        super().__init__(model, train_inputs)
        self._track_losses = {}
        self._training_loss = None
        self._hermes = Hermes(model)
        self._optimizer = None
        self._grads_and_vars = None

    def get_label_tensors(self):
        """
        Returns
        -------
        dict
            Contains pairs (tensor_name, tf.Tensor) of required tensors of labels.
        """
        if self._label_tensors is None:
            self._label_tensors = self._setup_label_placeholders()
        return self._label_tensors.copy()

    @abstractmethod
    def _setup_label_placeholders(self):
        """
        In case generator tensors are not provided, tf.placeholders will be used instead.

        Returns
        -------
        dict
            Contains pairs (tensor_name, tf.Tensor) of required tensors of labels.
        """
        pass

    def get_hermes(self):
        return self._hermes

    def compile(self):
        """
        Builds the training graph and the training loss.
        """
        super().compile()
        self.build_loss()
        self._hermes.setup_tensorboard()

    def build_loss(self):
        # noinspection PyAttributeOutsideInit
        loss = self._build_loss()
        self._training_loss = super()._build_final_loss(loss)
        assert self._training_loss is not None, '_build_loss method returned None, but must return the loss scalar.'
        self.track_loss(self._training_loss, Athena.TRAINING_LOSS)
        loss_is_built()

    @abstractmethod
    def _build_loss(self):
        # Must return the training loss scalar
        pass

    def track_loss(self, loss_tensor, loss_name):
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
        for i in range(epochs):
            it = tqdm(range(iter))

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
        print(input_feed_dict)
        print(label_feed_dict)
        for i in range(epochs):
            it = tqdm(range(iter))

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
                print(packed_data)
                print(packed_labels)
                feed_dict = packed_data.update(packed_labels)
                assert feed_dict is not None
                print(feed_dict)
                tracked_losses_vals, summary, _ = sess.run(
                    [track_losses, total_summary, train_op],
                    feed_dict=feed_dict
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

    @abstractmethod
    def get_label_feed_dict_config(self):
        """
        Returns
        -------
        dict
            Same as the input feed dict config, except it is for tensors with labels.
        """
        pass
