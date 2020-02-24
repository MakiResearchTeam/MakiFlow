from ..main_modules import ClassificatorBasis
from makiflow.base.loss_builder import Loss
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


class FocalTrainingModule(ClassificatorBasis):
    def _prepare_training_vars(self):
        self._focal_loss_is_built = False
        super()._prepare_training_vars()

    def _build_focal_loss(self):
        self._focal_loss = Loss.focal_loss(
            flattened_logits=self._logits,
            flattened_labels=self._labels,
            focal_num_positives=tf.ones_like(self._labels, dtype=tf.float32),
            num_classes=self._num_classes,
            focal_gamma=self._focal_gamma,
            ce_loss=self._ce_loss
        )
        self._final_focal_loss = super()._build_final_loss(self._focal_loss)

    def _setup_focal_loss_inputs(self):
        self._focal_gamma = tf.placeholder(tf.float32, shape=[], name='focal_gamma')

    def _minimize_focal_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._focal_loss_is_built:
            self._setup_focal_loss_inputs()
            self._build_focal_loss()
            self._focal_optimizer = optimizer
            self._focal_train_op = optimizer.minimize(
                self._final_focal_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))
            self._focal_loss_is_built = True

        if self._focal_optimizer != optimizer:
            print('New optimizer is used.')
            self._focal_optimizer = optimizer
            self._focal_train_op = optimizer.minimize(
                self._final_focal_loss, var_list=super()._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._focal_train_op

    # noinspection PyPep8Naming
    def fit_focal(
            self, Xtrain, Ytrain, Xtest, Ytest, optimizer=None, gamma=2.0, epochs=1, test_period=1, global_step=None
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
            gamma : float
                Parameter for the Focal Loss.
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
        assert (type(gamma) == float)
        assert (optimizer is not None)
        assert (self._session is not None)
        train_op = self._minimize_focal_loss(optimizer, global_step)
        # For testing
        Yish_test = tf.nn.softmax(self._inference_out)

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
                        [self._logits, self._final_focal_loss, train_op],
                        feed_dict={
                            self._images: Xbatch,
                            self._labels: Ybatch,
                            self._focal_gamma: gamma
                        }
                    )
                    # Use exponential decay for calculating loss and error
                    train_cost = 0.99 * train_cost + 0.01 * train_cost_batch
                    train_error_batch = error_rate(np.argmax(y_ish, axis=1), Ybatch)
                    train_error = 0.99 * train_error + 0.01 * train_error_batch

                # Validating the network on test data
                if i % test_period == 0:
                    # For test data
                    test_cost = np.float32(0)
                    test_predictions = np.zeros(len(Xtest))

                    for k in range(len(Xtest) // self._batch_sz):
                        # Test data
                        Xtestbatch = Xtest[k * self._batch_sz:(k + 1) * self._batch_sz]
                        Ytestbatch = Ytest[k * self._batch_sz:(k + 1) * self._batch_sz]
                        Yish_test_done = self._session.run(
                            Yish_test,
                            feed_dict={
                                self._images: Xtestbatch
                            }
                        ) + EPSILON
                        test_cost += sparse_cross_entropy(Yish_test_done, Ytestbatch)
                        test_predictions[k * self._batch_sz:(k + 1) * self._batch_sz] = np.argmax(Yish_test_done, axis=1)

                    # Collect and print data
                    test_cost = test_cost / (len(Xtest) // self._batch_sz)
                    test_error = error_rate(test_predictions, Ytest)
                    test_errors.append(test_error)
                    test_costs.append(test_cost)

                    train_costs.append(train_cost)
                    train_errors.append(train_error)

                    print('Epoch:', i, 'Train accuracy: {:0.4f}'.format(1 - train_error),
                          'Train cost: {:0.4f}'.format(train_cost),
                          'Test accuracy: {:0.4f}'.format(1 - test_error), 'Test cost: {:0.4f}'.format(test_cost))
        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()
            return {'train costs': train_costs, 'train errors': train_errors,
                    'test costs': test_costs, 'test errors': test_errors}
