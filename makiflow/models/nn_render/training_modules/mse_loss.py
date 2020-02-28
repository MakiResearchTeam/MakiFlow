import tensorflow as tf
from ..main_modules import NeuralRenderBasis
from makiflow.base.loss_builder import Loss
from makiflow.models.common.utils import print_train_info, moving_average
from makiflow.models.common.utils import loss_is_built, new_optimizer_used
from sklearn.utils import shuffle
from tqdm import tqdm

MSE_LOSS = 'MSE LOSS'


class MseTrainingModule(NeuralRenderBasis):
    def _prepare_training_vars(self):
        self._mse_loss_is_build = False
        super()._prepare_training_vars()

    def _build_mse_loss(self):
        self._mse_loss = Loss.mse_loss(self._images, self._training_out)
        self._final_mse_loss = self._build_final_loss(self._mse_loss)

    def _setup_mse_loss_inputs(self):
        pass

    def _minimize_mse_loss(self, optimizer, global_step):
        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._mse_loss_is_build:
            self._setup_mse_loss_inputs()
            self._build_mse_loss()
            self._mse_optimizer = optimizer
            self._mse_train_op = optimizer.minimize(
                self._final_mse_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))
            self._mse_loss_is_build = True
            loss_is_built()

        if self._mse_optimizer != optimizer:
            new_optimizer_used()
            self._mse_optimizer = optimizer
            self._mse_train_op = optimizer.minimize(
                self._final_mse_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._mse_train_op

    def fit_mse(self, images, uv_maps, optimizer, epochs=1, global_step=None, shuffle_data=True):
        """
        Method for training the model.

        Parameters
        ----------
        images : list
            Training images.
        uv_maps : list
            Training uvmaps.
        optimizer : TensorFlow optimizer
            Model uses TensorFlow optimizers in order train itself.
        epochs : int
            Number of epochs.
        global_step
            Please refer to TensorFlow documentation about global step for more info.
        shuffle_data : bool
            Set to False if you don't want the data to be shuffled.

        Returns
        -------
        python dictionary
            Dictionary with all testing data(train error, train cost, test error, test cost)
            for each test period.
        """
        assert (optimizer is not None)
        assert (self._session is not None)

        train_op = self._minimize_mse_loss(optimizer, global_step)

        n_batches = len(images) // self._batch_sz
        mse_losses = []
        iterator = None
        try:
            for i in range(epochs):
                if shuffle_data:
                    images, uv_maps = shuffle(images, uv_maps)
                mse_loss = 0
                iterator = tqdm(range(n_batches))

                for j in iterator:
                    Ibatch = images[j * self._batch_sz:(j + 1) * self._batch_sz]
                    Lbatch = uv_maps[j * self._batch_sz:(j + 1) * self._batch_sz]
                    batch_mse_loss, _ = self._session.run(
                        [self._final_mse_loss, train_op],
                        feed_dict={
                            self._images: Ibatch,
                            self._uv_maps: Lbatch
                        })
                    # Use exponential decay for calculating loss and error
                    mse_loss = moving_average(mse_loss, batch_mse_loss, j)

                mse_losses.append(mse_loss)

                print_train_info(i, (MSE_LOSS, mse_loss))
        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()
            return {MSE_LOSS: mse_losses}

    def gen_fit_mse(self, optimizer, epochs=1, iterations=10, global_step=None):
        """
        Method for training the model.

        Parameters
        ----------
        optimizer : tensorflow optimizer
            Model uses tensorflow optimizers in order train itself.
        epochs : int
            Number of epochs.
        iterations : int
            Defines how long one epoch is. One operation is a forward pass
            using one batch.
        global_step
            Please refer to TensorFlow documentation about global step for more info.

        Returns
        -------
        python dictionary
            Dictionary with all testing data(train error, train cost, test error, test cost)
            for each test period.
        """
        assert (optimizer is not None)
        assert (self._session is not None)

        train_op = self._minimize_mse_loss(optimizer, global_step)

        mse_losses = []
        iterator = None
        try:
            for i in range(epochs):
                mse_loss = 0
                iterator = tqdm(range(iterations))

                for j in iterator:
                    batch_mse_loss, _ = self._session.run(
                        [self._final_mse_loss, train_op],)
                    # Use exponential decay for calculating loss and error
                    mse_loss = moving_average(mse_loss, batch_mse_loss, j)

                mse_losses.append(mse_loss)

                print_train_info(i, (MSE_LOSS, mse_loss))
        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()
            return {MSE_LOSS: mse_losses}
