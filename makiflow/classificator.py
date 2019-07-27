from __future__ import absolute_import
from makiflow.beta_layers import MakiTensor, MakiOperation, InputLayer
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
from makiflow.utils import error_rate, sparse_cross_entropy

EPSILON = np.float32(1e-37)


class Classificator:
    def __init__(self, input: InputLayer, output: MakiTensor, num_classes: int):
        self.X = input.get_data_tensor()
        self.batch_sz = input.get_shape()[0]

        self.output = output
        self.labels = tf.placeholder(tf.int32, shape=[self.batch_sz, num_classes], name='labels')

    def __collect_params(self):
        current_tensor = self.output
        self.params = []
        self.named_params_dict = {}

        layer = current_tensor.get_binded_layer()

        while layer is not None:
            self.params += layer.get_params()
            self.named_params_dict.update(layer.get_params_dict())
            top_name = current_tensor.get_last_tensor_name()
            tensors_dict = current_tensor.get_previous_tensors()
            current_tensor = tensors_dict[top_name]
            layer = current_tensor.get_binded_layer()

    def set_session(self, session: tf.Session):
        self.session = session
        init_op = tf.variables_initializer(self.params)
        self.session.run(init_op)

    def pure_fit(self, Xtrain, Ytrain, Xtest, Ytest, optimizer=None, epochs=1, test_period=1):
        """
        Method for training the model. Works faster than `verbose_fit` method because
        it uses exponential decay in order to speed up training. It produces less accurate
        train error measurement.

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
        assert (self.session is not None)
        # This is for correct working of tqdm loop. After KeyboardInterrupt it breaks and
        # starts to print progress bar each time it updates.
        # In order to avoid this problem we handle KeyboardInterrupt exception and close
        # the iterator tqdm iterates through manually. Yes, it's ugly, but necessary for
        # convinient working with MakiFlow in Jupyter Notebook. Sometimes it's helpful
        # even for console applications.
        iterator = None

        Xtrain = Xtrain.astype(np.float32)
        Xtest = Xtest.astype(np.float32)

        # For training
        cost = (
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.labels)),
            self.output)
        train_op = (cost, optimizer.minimize(cost[0]))
        # Initialize optimizer's variables
        self.session.run(tf.variables_initializer(optimizer.variables()))

        # For testing
        Yish_test = tf.nn.softmax(self.output)

        n_batches = Xtrain.shape[0] // self.batch_sz

        train_costs = []
        train_errors = []
        test_costs = []
        test_errors = []
        try:
            for i in range(epochs):
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
                train_cost = np.float32(0)
                train_error = np.float32(0)
                iterator = tqdm(range(n_batches))

                for j in iterator:
                    Xbatch = Xtrain[j * self.batch_sz:(j + 1) * self.batch_sz]
                    Ybatch = Ytrain[j * self.batch_sz:(j + 1) * self.batch_sz]

                    (train_cost_batch, y_ish), _ = self.session.run(
                        train_op,
                        feed_dict={self.X: Xbatch, self.output: Ybatch})
                    # Use exponential decay for calculating loss and error
                    train_cost = 0.99 * train_cost + 0.01 * train_cost_batch
                    train_error_batch = error_rate(np.argmax(y_ish, axis=1), Ybatch)
                    train_error = 0.99 * train_error + 0.01 * train_error_batch

                # Validating the network on test data
                if i % test_period == 0:
                    # For test data
                    test_cost = np.float32(0)
                    test_predictions = np.zeros(len(Xtest))

                    for k in range(len(Xtest) // self.batch_sz):
                        # Test data
                        Xtestbatch = Xtest[k * self.batch_sz:(k + 1) * self.batch_sz]
                        Ytestbatch = Ytest[k * self.batch_sz:(k + 1) * self.batch_sz]
                        Yish_test_done = self.session.run(Yish_test, feed_dict={self.X: Xtestbatch}) + EPSILON
                        test_cost += sparse_cross_entropy(Yish_test_done, Ytestbatch)
                        test_predictions[k * self.batch_sz:(k + 1) * self.batch_sz] = np.argmax(Yish_test_done, axis=1)

                    # Collect and print data
                    test_cost = test_cost / (len(Xtest) // self.batch_sz)
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


