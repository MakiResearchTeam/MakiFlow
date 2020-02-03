import tensorflow as tf
from ..main_modules import NeuralRendererBasis
from makiflow.base.loss_builder import Loss
from sklearn.utils import shuffle
from tqdm import tqdm


class MseTrainingModule(NeuralRendererBasis):
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

        if self._mse_optimizer != optimizer:
            print('New optimizer is used.')
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

        Returns
        -------
        python dictionary
            Dictionary with all testing data(train error, train cost, test error, test cost)
            for each test period.
        """
        assert (optimizer is not None)
        assert (self._session is not None)

        train_op = self._minimize_mse_loss(optimizer, global_step)

        n_batches = len(images) // self.batch_sz
        train_focal_losses = []
        iterator = None
        try:
            for i in range(epochs):
                if shuffle_data:
                    images, uv_maps = shuffle(images, uv_maps)
                abs_loss = 0
                iterator = tqdm(range(n_batches))

                for j in iterator:
                    Ibatch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                    Lbatch = uv_maps[j * self.batch_sz:(j + 1) * self.batch_sz]
                    batch_abs_loss, _ = self._session.run(
                        [self._final_mse_loss, train_op],
                        feed_dict={
                            self._images: Ibatch,
                            self._uv_maps: Lbatch
                        })
                    # Use exponential decay for calculating loss and error
                    abs_loss = 0.1 * batch_abs_loss + 0.9 * abs_loss

                train_focal_losses.append(abs_loss)

                print('Epoch:', i, 'Abs loss: {:0.4f}'.format(abs_loss))
        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()
            return {'train losses': train_focal_losses}
