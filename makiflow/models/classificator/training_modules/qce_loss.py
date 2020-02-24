from ..main_modules import ClassificatorBasis
from makiflow.base.loss_builder import Loss
from makiflow.models.common.utils import print_train_info, moving_average
from makiflow.models.common.utils import new_optimizer_used, loss_is_built
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm


TRAIN_ACCURACY = 'TRAIN ACCURACY'
TRAIN_LOSS = 'TRAIN LOSS'
TEST_ACCURACY = 'TEST ACCURACY'
TEST_LOSS = 'TEST LOSS'


class QCETrainingModule(ClassificatorBasis):
    def _prepare_training_vars(self):
        self._qce_loss_is_built = False
        super()._prepare_training_vars()

    def _build_quadratic_ce_loss(self):
        self._quadratic_ce_loss = Loss.quadratic_ce_loss(
            ce_loss=self._ce_loss
        )
        self._final_quadratic_ce_loss = super()._build_final_loss(self._quadratic_ce_loss)

    def _setup_quadratic_ce_loss_inputs(self):
        pass

    def _minimize_quadratic_ce_loss(self, optimizer, global_step):
        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._qce_loss_is_built:
            self._setup_quadratic_ce_loss_inputs()
            self._build_quadratic_ce_loss()
            self._quadratic_ce_optimizer = optimizer
            self._quadratic_ce_train_op = optimizer.minimize(
                self._final_quadratic_ce_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))
            self._qce_loss_is_built = False
            loss_is_built()

        if self._quadratic_ce_optimizer != optimizer:
            new_optimizer_used()
            self._quadratic_ce_optimizer = optimizer
            self._quadratic_ce_train_op = optimizer.minimize(
                self._final_quadratic_ce_loss, var_list=super()._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._quadratic_ce_train_op

    # noinspection PyPep8Naming
    def fit_quadratic_ce(
            self, Xtrain, Ytrain, Xtest, Ytest, optimizer=None, epochs=1, test_period=1,
            global_step=None
    ):
        """
        Method for training the model.

        Parameters
        ----------
        Xtrain : numpy array
            Training images stacked into one big array with shape (num_images, image_w, image_h, image_depth).
        Ytrain : numpy array
            Training label for each image in `Xtrain` array with shape (num_images).
            IMPORTANT: ALL LABELS MUST BE NOT ONE-HOT ENCODED, USE SPARSE TRAINING DATA INSTEAD.
        Xtest : numpy array
            Same as `Xtrain` but for testing.
        Ytest : numpy array
            Same as `Ytrain` but for testing.
        optimizer : tensorflow optimizer
            Model uses tensorflow optimizers in order train itself.
        epochs : int
            Number of epochs.
        test_period : int
            Test begins each `test_period` epochs. You can set a larger number in order to
            speed up training.

        Returns
        -------
            python dictionary
                Dictionary with all testing data(train error, train cost, test error, test cost)
                for each test period.
        """
        assert (optimizer is not None)
        assert (self._session is not None)
        train_op = self._minimize_quadratic_ce_loss(optimizer, global_step)

        n_batches = len(Xtrain) // self._batch_sz

        train_costs = []
        train_errors = []
        test_costs = []
        test_errors = []
        iterator = None
        try:
            for i in range(epochs):
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
                train_cost = np.float32(0)
                train_error = np.float32(0)
                iterator = tqdm(range(n_batches))

                for j in iterator:
                    Xbatch = Xtrain[j * self._batch_sz:(j + 1) * self._batch_sz]
                    Ybatch = Ytrain[j * self._batch_sz:(j + 1) * self._batch_sz]
                    y_ish, train_cost_batch, _ = self._session.run(
                        [self._logits, self._final_quadratic_ce_loss, train_op],
                        feed_dict={
                            self._images: Xbatch,
                            self._labels: Ybatch
                        }
                    )
                    # Use exponential decay for calculating loss and error
                    train_cost = 0.99 * train_cost + 0.01 * train_cost_batch

                train_costs.append(train_cost)
                train_errors.append(train_error)
                train_info = [(TRAIN_ACCURACY, 1 - train_error), (TRAIN_LOSS, train_cost)]
                # Validating the network on test data
                if test_period != -1 and i % test_period == 0:
                    # For test data
                    test_error, test_cost = self.evaluate(Xtest, Ytest)
                    test_errors.append(test_error)
                    test_costs.append(test_cost)
                    train_info += [(TEST_ACCURACY, 1 - test_error), (TRAIN_LOSS, test_cost)]

                print_train_info(i, *train_info)
        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()
            return {'train costs': train_costs, 'train errors': train_errors,
                    'test costs': test_costs, 'test errors': test_errors}
