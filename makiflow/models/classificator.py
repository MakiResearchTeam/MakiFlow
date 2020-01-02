from __future__ import absolute_import
from makiflow.base import MakiModel, MakiTensor, Loss
from makiflow.layers import InputLayer
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
from makiflow.utils import error_rate, sparse_cross_entropy
from copy import copy

EPSILON = np.float32(1e-37)


class Classificator(MakiModel):

    def __init__(self, input: InputLayer, output: MakiTensor, name='MakiClassificator'):
        graph_tensors = copy(output.get_previous_tensors())
        # Add output tensor to `graph_tensors` since it doesn't have it.
        # It is assumed that graph_tensors contains ALL THE TENSORS graph consists of.
        graph_tensors.update(output.get_self_pair())
        outputs = [output]
        inputs = [input]
        super().__init__(graph_tensors, outputs, inputs)
        self.name = str(name)
        self._batch_sz = input.get_shape()[0]
        self._images = self._input_data_tensors[0]
        self._inference_out = self._output_data_tensors[0]
        # For training
        self._training_vars_are_ready = False

    def _get_model_info(self):
        input_mt = self._inputs[0]
        output_mt = self._outputs[0]
        return {
            'input_mt': input_mt.get_name(),
            'output_mt': output_mt.get_name(),
            'name': self.name
        }

    def _prepare_training_vars(self):
        if not self._set_for_training:
            super()._setup_for_training()

        self._logits = self._training_outputs[0]
        self._num_classes = self._logits.get_shape()[-1]
        self._labels = tf.placeholder(tf.int32, shape=[self._batch_sz])
        self._ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self._logits, labels=self._labels
        )
        self._training_vars_are_ready = True
        self._ce_loss_is_build = False
        self._focal_loss_is_build = False
        self._maki_loss_is_build = False

    def _build_ce_loss(self):
        ce_loss = tf.reduce_mean(self._ce_loss)
        self._final_ce_loss = self._build_final_loss(ce_loss)

        self._ce_loss_is_build = True

    def _minimize_ce_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._ce_loss_is_build:
            # no need to setup any inputs for this loss
            self._build_ce_loss()
            self._ce_optimizer = optimizer
            self._ce_train_op = optimizer.minimize(
                self._final_ce_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._ce_optimizer != optimizer:
            print('New optimizer is used.')
            self._ce_optimizer = optimizer
            self._ce_train_op = optimizer.minimize(
                self._final_ce_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._ce_train_op

    def fit_ce(
            self, Xtrain, Ytrain, Xtest, Ytest, optimizer=None, epochs=1, test_period=1, global_step=None
    ):
        """
        Method for training the model. Works faster than `verbose_fit` method because
        it uses exponential decay in order to speed up training. It produces less accurate
        train error mesurement.

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
        train_op = self._minimize_ce_loss(optimizer, global_step)
        # For testing
        Yish_test = tf.nn.softmax(self._inference_out)

        n_batches = Xtrain.shape[0] // self._batch_sz

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
                        [self._logits, self._final_ce_loss, train_op],
                        feed_dict={self._images: Xbatch, self._labels: Ybatch})
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
                        Yish_test_done = self._session.run(Yish_test, feed_dict={self._images: Xtestbatch}) + EPSILON
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
        finally:
            if iterator is not None:
                iterator.close()
            return {'train costs': train_costs, 'train errors': train_errors,
                    'test costs': test_costs, 'test errors': test_errors}

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------FOCAL LOSS--------------------------------------------------

    def _build_focal_loss(self):
        self._focal_loss = Loss.focal_loss(
            flattened_logits=self._logits,
            flattened_labels=self._labels,
            num_positives=tf.ones_like(self._labels, dtype=tf.float32),
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

        if not self._focal_loss_is_build:
            self._setup_focal_loss_inputs()
            self._build_focal_loss()
            self._focal_optimizer = optimizer
            self._focal_train_op = optimizer.minimize(
                self._final_focal_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

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

        n_batches = Xtrain.shape[0] // self._batch_sz

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
        finally:
            if iterator is not None:
                iterator.close()
            return {'train costs': train_costs, 'train errors': train_errors,
                    'test costs': test_costs, 'test errors': test_errors}

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------MAKI LOSS---------------------------------------------------

    def _build_maki_loss(self, gamma):
        self._maki_loss = Loss.maki_loss(
            flattened_logits=self._logits,
            flattened_labels=self._labels,
            num_positives=tf.ones_like(self._labels, dtype=tf.float32),
            num_classes=self._num_classes,
            maki_gamma=gamma,
            ce_loss=self._ce_loss
        )
        self._final_maki_loss = super()._build_final_loss(self._maki_loss)

    def _setup_maki_loss_inputs(self):
        pass

    def _minimize_maki_loss(self, optimizer, global_step, gamma):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._maki_loss_is_build:
            self._setup_maki_loss_inputs()
            self._build_maki_loss(gamma)
            self._maki_optimizer = optimizer
            self._maki_train_op = optimizer.minimize(
                self._final_maki_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._maki_optimizer != optimizer:
            print('New optimizer is used.')
            self._maki_optimizer = optimizer
            self._maki_train_op = optimizer.minimize(
                self._final_maki_loss, var_list=super()._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._maki_train_op

    # noinspection PyPep8Naming
    def fit_maki(
            self, Xtrain, Ytrain, Xtest, Ytest, optimizer=None, gamma=2, epochs=1, test_period=1, global_step=None
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
        assert (type(gamma) == int)
        assert (optimizer is not None)
        assert (self._session is not None)
        train_op = self._minimize_maki_loss(optimizer, global_step, gamma)
        # For testing
        Yish_test = tf.nn.softmax(self._inference_out)

        n_batches = Xtrain.shape[0] // self._batch_sz

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
                        [self._logits, self._final_maki_loss, train_op],
                        feed_dict={
                            self._images: Xbatch,
                            self._labels: Ybatch
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
        finally:
            if iterator is not None:
                iterator.close()
            return {'train costs': train_costs, 'train errors': train_errors,
                    'test costs': test_costs, 'test errors': test_errors}

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------QUADRATIC CE LOSS-------------------------------------------

    def _build_quadratic_ce_loss(self):
        self._quadratic_ce_loss = Loss.quadratic_ce_loss(
            ce_loss=self._ce_loss
        )
        self._final_quadratic_ce_loss = super()._build_final_loss(self._quadratic_ce_loss)

    def _setup_quadratic_ce_loss_inputs(self):
        pass

    def _minimize_quadratic_ce_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._maki_loss_is_build:
            self._setup_quadratic_ce_loss_inputs()
            self._build_quadratic_ce_loss()
            self._quadratic_ce_optimizer = optimizer
            self._quadratic_ce_train_op = optimizer.minimize(
                self._final_quadratic_ce_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._quadratic_ce_optimizer != optimizer:
            print('New optimizer is used.')
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
        # For testing
        Yish_test = tf.nn.softmax(self._inference_out)

        n_batches = Xtrain.shape[0] // self._batch_sz

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
                        test_predictions[k * self._batch_sz:(k + 1) * self._batch_sz] = np.argmax(
                            Yish_test_done, axis=1)

                    # Collect and print data
                    test_cost = test_cost / (len(Xtest) // self._batch_sz)
                    test_error = error_rate(test_predictions, Ytest)
                    test_errors.append(test_error)
                    test_costs.append(test_cost)

                    train_costs.append(train_cost)
                    train_errors.append(train_error)

                    print('Epoch:', i, 'Train accuracy: {:0.4f}'.format(1 - train_error),
                          'Train cost: {:0.4f}'.format(train_cost),
                          'Test accuracy: {:0.4f}'.format(1 - test_error),
                          'Test cost: {:0.4f}'.format(test_cost))
        except Exception as ex:
            print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {'train costs': train_costs, 'train errors': train_errors,
                    'test costs': test_costs, 'test errors': test_errors}

    def evaluate(self, Xtest, Ytest, batch_sz):
        # TODO: for test can be delete
        # Validating the network
        Xtest = Xtest.astype(np.float32)
        Yish_test = tf.nn.softmax(self._inference_out)
        n_batches = Xtest.shape[0] // batch_sz

        # For train data
        test_cost = 0
        predictions = np.zeros(len(Xtest))
        for k in tqdm(range(n_batches)):
            # Test data
            Xtestbatch = Xtest[k * batch_sz:(k + 1) * batch_sz]
            Ytestbatch = Ytest[k * batch_sz:(k + 1) * batch_sz]
            Yish_test_done = self._session.run(Yish_test, feed_dict={self._images: Xtestbatch}) + EPSILON
            test_cost += sparse_cross_entropy(Yish_test_done, Ytestbatch)
            predictions[k * batch_sz:(k + 1) * batch_sz] = np.argmax(Yish_test_done, axis=1)

        error = error_rate(predictions, Ytest)
        test_cost = test_cost / (len(Xtest) // batch_sz)
        print('Accuracy:', 1 - error, 'Cost:', test_cost)


