# For saving the architecture
import json
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tqdm import tqdm

from .utils import error_rate, sparse_cross_entropy

# For testing and evaluating the model
EPSILON = np.float32(1e-37)


class ConvModel(object):
    def __init__(self, layers, input_shape, num_classes, name='MakiModel'):
        """
        Parameters
        ----------
        layers : list
            List of layers the model consists of.
            Example:
            [   ConvLayer(5, 5, 3, 32, stride=1, padding='valid'),
                FlattenLayer(),
                DenseLayer(...)
            ]
        input_shape : list
            List represents input shape: [batch size, input width, input height, input depth].
        Example: [64, 224, 224, 3]
        :param output_shape - tuple represents output shape: (batch size, number of classes)
        ATTENTION: DONT ADD SOFTMAX ACTIVATION FUNCTION AT THE END,
        CONVMODEL DOES IT ITSELF. IT'S NEEDED FOR TRAINING OPERATION
        """

        self.name = name
        self.input_shape = list(input_shape)
        self.num_classes = num_classes

        self.batch_sz = input_shape[0]
        self.layers = layers
        self.session = None

        self.params = []
        for layer in self.layers:
            self.params += layer.get_params()

        # Get params and store them into python dictionary in order to save and load them correctly later
        self.named_params_dict = {}
        for layer in self.layers:
            self.named_params_dict.update(layer.get_params_dict())

        # Used for training
        self.X = tf.placeholder(tf.float32, shape=input_shape)
        self.sparse_out = tf.placeholder(tf.int32, shape=self.batch_sz)

        # Used for actual running
        self.input = tf.placeholder(tf.float32, shape=(input_shape))
        self.output = tf.nn.softmax(self.forward(self.input))

        
    def set_session(self, session):
        self.session = session
        # for layer in self.layers:
        #    layer.session = session

        init_op = tf.variables_initializer(self.params)
        self.session.run(init_op)

        
    def predict(self, X):
        assert (self.session is not None)
        return self.session.run(
            self.output,
            feed_dict={self.input: X}
        )

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    
    def forward_train(self, X):
        for layer in self.layers:
            X = layer.forward(X, is_training=True)
        return X

    
    def save_weights(self, path):
        """
        This function uses default TensorFlow's way for saving models - checkpoint files.
        :param path - full path+name of the model.
        Example: '/home/student401/my_model/model.ckpt'
        """
        assert (self.session is not None)
        saver = tf.train.Saver(self.named_params_dict)
        save_path = saver.save(self.session, path)
        print('Model saved to %s' % save_path)

        
    def load_weights(self, path):
        """
        This function uses default TensorFlow's way for restoring models - checkpoint files.
        :param path - full path+name of the model.
        Example: '/home/student401/my_model/model.ckpt'
        """
        assert (self.session is not None)
        saver = tf.train.Saver(self.named_params_dict)
        saver.restore(self.session, path)
        print('Model restored')

        
    def to_json(self, path):
        """
        Convert model's architecture to json.json file and save it.
        path - path to file to save in.
        """

        model_dict = {
            'name': self.name,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }
        layers_dict = {
            'layers': []
        }
        for layer in self.layers:
            layers_dict['layers'].append(layer.to_dict())

        model_dict.update(layers_dict)
        model_json = json.dumps(model_dict, indent=1)
        json_file = open(path, mode='w')
        json_file.write(model_json)
        json_file.close()
        print("Model's architecture is saved to {}.".format(path))

        
    def evaluate(self, Xtest, Ytest):
        # TODO: n_batches is never used
        # Validating the network
        Xtest = Xtest.astype(np.float32)
        Ytest = Ytest.argmax(axis=1)

        Yish_test = tf.nn.softmax(self.forward(self.X))
        n_batches = Xtest.shape[0] // self.batch_sz

        # For train data
        test_cost = 0
        predictions = np.zeros(len(Xtest))
        for k in tqdm(range(n_batches)):
            # Test data
            Xtestbatch = Xtest[k * self.batch_sz:(k + 1) * self.batch_sz]
            Ytestbatch = Ytest[k * self.batch_sz:(k + 1) * self.batch_sz]
            Yish_test_done = self.session.run(Yish_test, feed_dict={self.X: Xtestbatch}) + EPSILON
            test_cost += sparse_cross_entropy(Yish_test_done, Ytestbatch)
            predictions[k * self.batch_sz:(k + 1) * self.batch_sz] = np.argmax(Yish_test_done, axis=1)

        error = error_rate(predictions, Ytest)
        test_cost = test_cost / (len(Xtest) // self.batch_sz)
        print('Accuracy:', 1 - error, 'Cost:', test_cost)

        
    def verbose_fit(self, Xtrain, Ytrain, Xtest, Ytest, optimizer=None, epochs=1, test_period=1):
        """
        Method for training the model. Works slower than `verbose_fit` method because
        it computes error and cost both for train and test data. It produces most accurate
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
        assert (self.session is not None)
        Xtrain = Xtrain.astype(np.float32)
        Xtest = Xtest.astype(np.float32)

        # For training
        Yish = self.forward_train(self.X)
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Yish, labels=self.sparse_out))
        train_op = optimizer.minimize(cost)
        # Initilize optimizer's variables
        self.session.run(tf.variables_initializer(optimizer.variables()))

        # For testing
        Yish_test = tf.nn.softmax(self.forward(self.X))

        n_batches = Xtrain.shape[0] // self.batch_sz

        train_costs = []
        train_errors = []
        test_costs = []
        test_errors = []
        for i in range(epochs):
            Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
            for j in tqdm(range(n_batches)):
                Xbatch = Xtrain[j * self.batch_sz:(j + 1) * self.batch_sz]
                Ybatch = Ytrain[j * self.batch_sz:(j + 1) * self.batch_sz]

                self.session.run(train_op, feed_dict={self.X: Xbatch, self.sparse_out: Ybatch})

            # Validating the network
            if i % test_period == 0:
                # Validating the network on test data
                test_cost = 0
                test_predictions = np.zeros(len(Xtest))
                for k in range(len(Xtest) // self.batch_sz):
                    Xtestbatch = Xtest[k * self.batch_sz:(k + 1) * self.batch_sz]
                    Ytestbatch = Ytest[k * self.batch_sz:(k + 1) * self.batch_sz]
                    Yish_test_done = self.session.run(Yish_test, feed_dict={self.X: Xtestbatch}) + EPSILON
                    test_cost += sparse_cross_entropy(Yish_test_done, Ytestbatch)
                    test_predictions[k * self.batch_sz:(k + 1) * self.batch_sz] = np.argmax(Yish_test_done, axis=1)

                # Validating the network on train data    
                train_cost = 0
                train_predictions = np.zeros(len(Xtrain))
                for k in range(len(Xtrain) // self.batch_sz):
                    Xtrainbatch = Xtrain[k * self.batch_sz:(k + 1) * self.batch_sz]
                    Ytrainbatch = Ytrain[k * self.batch_sz:(k + 1) * self.batch_sz]
                    Yish_train_done = self.session.run(Yish_test, feed_dict={self.X: Xtrainbatch}) + EPSILON
                    train_cost += sparse_cross_entropy(Yish_train_done, Ytrainbatch)
                    train_predictions[k * self.batch_sz:(k + 1) * self.batch_sz] = np.argmax(Yish_train_done, axis=1)

                # Normalize cost values so that we're able to compare them while training
                test_cost =  test_cost / (len(Xtest) // self.batch_sz)
                train_cost =  train_cost / (len(Xtrain) // self.batch_sz)
                
                test_error = error_rate(test_predictions, Ytest)
                train_error = error_rate(train_predictions, Ytrain)

                train_errors.append(train_error)
                train_costs.append(train_cost)

                test_errors.append(test_error)
                test_costs.append(test_cost)

                print('Epoch:', i, 'Train accuracy: {:0.4f}'.format(1 - train_error), 'Train cost: {:0.4f}'.format(train_cost),
                      'Test accuracy: {:0.4f}'.format(1 - test_error), 'Test cost: {:0.4f}'.format(test_cost))

        return {'train costs': train_costs, 'train errors': train_errors,
                'test costs': test_costs, 'test errors': test_errors}

    def pure_fit(self, Xtrain, Ytrain, Xtest, Ytest, optimizer=None, epochs=1, test_period=1):
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
        assert (self.session is not None)
        # This is for correct working of tqdm loop. After KeyboardInterrupt it breaks and
        # starts to print progress bar each time it updates.
        # In order to avoid this problem we handle KeyboardInterrupt exception and close
        # the iterator tqdm iterates through manually. Yes, it's ugly, but necessary for
        # convinient working with MakiFlow in Jupyter Notebook. Sometimes it's helpful
        # even for console applications.
        try:
            Xtrain = Xtrain.astype(np.float32)
            Xtest = Xtest.astype(np.float32)

            # For training
            Yish = self.forward_train(self.X)
            cost = (
                tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Yish, labels=self.sparse_out)), Yish)
            train_op = (cost, optimizer.minimize(cost[0]))
            # Initialize optimizer's variables
            self.session.run(tf.variables_initializer(optimizer.variables()))

            # For testing
            Yish_test = tf.nn.softmax(self.forward(self.X))

            n_batches = Xtrain.shape[0] // self.batch_sz

            train_costs = []
            train_errors = []
            test_costs = []
            test_errors = []
            for i in range(epochs):
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
                train_cost = np.float32(0)
                train_error = np.float32(0)
                iterator = range(n_batches)

                for j in tqdm(iterator):
                    Xbatch = Xtrain[j * self.batch_sz:(j + 1) * self.batch_sz]
                    Ybatch = Ytrain[j * self.batch_sz:(j + 1) * self.batch_sz]

                    (train_cost_batch, y_ish), _ = self.session.run(train_op,
                                                                    feed_dict={self.X: Xbatch, self.sparse_out: Ybatch})
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

                    print('Epoch:', i, 'Train accuracy:', 1 - train_error, 'Train cost:', train_cost,
                          'Test accuracy', 1 - test_error, 'Test cost', test_cost)
        except KeyboardInterrupt:
            iterator.close() 
        finally:
                  

            return {'train costs': train_costs, 'train errors': train_errors,
                'test costs': test_costs, 'test errors': test_errors}
