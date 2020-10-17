import tensorflow as tf
from makiflow.models.common.utils import print_train_info, moving_average
from makiflow.models.common.utils import new_optimizer_used, loss_is_built
from .core import TrainingCore
from tqdm import tqdm
from abc import abstractmethod
from .hermes import Hermes


class Athena(TrainingCore):
    # Athena is the goddess of wisdom and intelligence. Athena is the one who trains your model.
    TRAINING_LOSS = 'TRAINING_LOSS'

    def _setup_for_training(self):
        super()._setup_for_training()
        self._track_losses = {}
        self._training_loss = None
        self._hermes = Hermes(super().get_model())

    def get_hermes(self):
        return self._hermes

    def track_loss(self, loss_tensor, loss_name):
        loss = self._track_losses.get(loss_name)
        if loss is not None:
            print(f'Overriding already existing {loss_name} loss tensor.')

        self._track_losses[loss_name] = loss_tensor
        self._hermes.add_scalar(loss, loss_name)

    def compile(self):
        """
        Builds the training graph and the training loss.
        """
        super().compile()
        self.build_loss()
        self._hermes.setup_tensorboard()

    def build_loss(self):
        # noinspection PyAttributeOutsideInit
        self._training_loss = self._build_loss()
        assert self._training_loss is not None, '_build_loss method returned None, but must return the loss scalar.'
        self.track_loss(self._training_loss, Athena.TRAINING_LOSS)
        loss_is_built()

    def get_track_losses(self):
        return self._track_losses.copy()

    @abstractmethod
    def _build_loss(self):
        # Must return the training loss scalar
        pass


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
        train_inputs = self.get
        for i in range(epochs):
            it = tqdm(range(iter))

            # Loss value holders. They will hold an interpolated loss value for one iteration.
            # This loss value will then be passed to an appropriate loss value collector.
            loss_holders = {}
            for loss_name in self.get_track_losses():
                loss_holders[loss_name] = 0.0

            # Performs training iterations
            for j in it:
                data = next(generator)
                self
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

    def __minimize_loss(self, optimizer, global_step):
        assert optimizer is not None, 'No optimizer is provided.'
        assert super().is_compiled(), 'The model is not compiled.'

        if self._optimizer != optimizer:
            self.__create_train_op(optimizer, global_step)

        return self._train_op

    def __create_train_op(self, optimizer, global_step):
        self._optimizer = optimizer

        if self._grads_and_vars is None:
            training_vars = self._model.get_training_vars()
            # Returns list of tuples: [ (grad, var) ]
            self._grads_and_vars = optimizer.compute_gradients(self._training_loss, training_vars)
            vars_and_grads = [(var, grad) for grad, var in self._grads_and_vars]
            self._hermes.set_vars_grads(vars_and_grads)

        self._train_op = optimizer.apply_gradients(
            grads_and_vars=self._grads_and_vars, global_step=global_step
        )

        self.get_session().run(tf.variables_initializer(optimizer.variables()))
        new_optimizer_used()
