from .layers import Layer
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from .utils import error_rate, cross_entropy, sparse_cross_entropy
from tqdm import tqdm

# For saving the architecture
import json

# For testing and evaluating the model
EPSILON = np.float32(1e-37)

class ConvModel(object):
    def __init__(self, layers, input_shape, output_shape, name='MakiModel'):
        """
        layers - list of layers model consists of. Example:
        [   ConvLayer(5, 5, 3, 32, stride=1, padding='valid'),
            FlattenLayer(),
            DenseLayer(...)
        ]
        input_shape - tuple represents input shape: (batch size, input width, input height, input depth).
        Example: [64, 224, 224, 3]
        output_shape - tuple represents output shape: (batch size, number of classes)
        
        ATTENTION: DONT ADD SOFTMAX ACTIVATION FUNCTION AT THE END,
        CONVMODEL DOES IT ITSELF. IT'S NEEDED FOR TRAINING OPERATION
        """
        
        self.name = name
        self.input_shape = list(input_shape)
        self.output_shape = list(output_shape)
        
        self.batch_sz = input_shape[0]
        self.layers = layers
        
        
        self.params = []
        for layer in self.layers:
            self.params += layer.get_params()
            
        # Get params and store them into python dictionary in order to save and load them correctly later
        self.named_params_dict = {}
        for layer in self.layers:
            self.named_params_dict.update(layer.get_params_dict())
        
        # Used for training
        self.X = tf.placeholder(tf.float32, shape=input_shape)
        self.sparse_out = tf.placeholder(tf.int32, shape=input_shape[0])
        
        # Used for actual running
        self.input = tf.placeholder(tf.float32, shape=(None, *input_shape[1:]))
        self.output = tf.nn.softmax(self.forward(self.input))
    
    
    def set_session(self, session):
        self.session = session
        #for layer in self.layers:
        #    layer.session = session
            
        init_op = tf.variables_initializer(self.params)
        self.session.run(init_op)
                
        
    def predict(self, X):
        assert(self.session is not None)
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
        path - full path+name of the model. 
        Example: '/home/student401/my_model/model.ckpt'
        """
        assert(self.session is not None)
        saver = tf.train.Saver(self.named_params_dict)
        save_path = saver.save(self.session, path)
        print('Model saved to %s' % save_path)
    
    
    def load_weights(self, path):
        """
        This function uses default TensorFlow's way for restoring models - checkpoint files.
        path - full path+name of the model. 
        Example: '/home/student401/my_model/model.ckpt'
        """
        assert(self.session is not None)
        saver = tf.train.Saver(self.named_params_dict)
        saver.restore(self.session, path)
        print('Model restored')
        
    
    def to_json(self, path):
        """
        Convert model's architecture to json file and save it.
        path - path to file to save in.
        """
        
        model_dict = {
            'name': self.name,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape
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
        # Validating the network
        Xtest = Xtest.astype(np.float32)
        Ytest = Ytest.argmax(axis=1)
        
        Yish_test = tf.nn.softmax(self.forward(self.X))
        n_batches = Xtest.shape[0] // self.batch_sz
        
        # For train data
        test_cost = 0
        predictions = np.zeros(len(Xtest))
        for k in tqdm(range(len(Xtest) // self.batch_sz)):
            # Test data
            Xtestbatch = Xtest[k*self.batch_sz:(k+1)*self.batch_sz]
            Ytestbatch = Ytest[k*self.batch_sz:(k+1)*self.batch_sz]
            Yish_test_done = self.session.run(Yish_test, feed_dict={self.X: Xtestbatch}) + EPSILON
            test_cost += sparse_cross_entropy(Yish_test_done, Ytestbatch)
            predictions[k*self.batch_sz:(k+1)*self.batch_sz] = np.argmax(Yish_test_done, axis=1)

        error = error_rate(predictions, Ytest)
        test_cost = test_cost / (len(Xtest) // self.batch_sz)
        print('Accuracy:', 1 - error, 'Cost:', test_cost)
        
        
    def verbose_fit(self, Xtrain, Ytrain, Xtest, Ytest, optimizer=None, epochs=1, test_period=1):
        assert(optimizer is not None)
        assert(self.session is not None)
        Xtrain = Xtrain.astype(np.float32)
        Ytrain = Ytrain.argmax(axis=1)
        Xtest = Xtest.astype(np.float32)
        Ytest = Ytest.argmax(axis=1)
        
        # For training
        Yish = self.forward_train(self.X)
        cost = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Yish, labels=self.sparse_out) )
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
                Xbatch = Xtrain[j*self.batch_sz:(j+1)*self.batch_sz]
                Ybatch = Ytrain[j*self.batch_sz:(j+1)*self.batch_sz]

                self.session.run(train_op, feed_dict={self.X: Xbatch, self.sparse_out: Ybatch})
                
            # Validating the network
            if i % test_period == 0:
                # Validating the network on test data
                test_cost = 0
                test_predictions = np.zeros(len(Xtest))
                for k in range(len(Xtest) // self.batch_sz):
                    Xtestbatch = Xtest[k*self.batch_sz:(k+1)*self.batch_sz]
                    Ytestbatch = Ytest[k*self.batch_sz:(k+1)*self.batch_sz]
                    Yish_test_done = self.session.run(Yish_test, feed_dict={self.X: Xtestbatch}) + EPSILON
                    test_cost += sparse_cross_entropy(Yish_test_done, Ytestbatch)
                    test_predictions[k*self.batch_sz:(k+1)*self.batch_sz] = np.argmax(Yish_test_done, axis=1)

                # Validating the network on train data    
                train_cost = 0
                train_predictions = np.zeros(len(Xtrain))
                for k in range(len(Xtrain) // self.batch_sz):
                    Xtrainbatch = Xtrain[k*self.batch_sz:(k+1)*self.batch_sz]
                    Ytrainbatch = Ytrain[k*self.batch_sz:(k+1)*self.batch_sz]
                    Yish_train_done = self.session.run(Yish_test, feed_dict={self.X: Xtrainbatch}) + EPSILON
                    train_cost += sparse_cross_entropy(Yish_train_done, Ytrainbatch)
                    train_predictions[k*self.batch_sz:(k+1)*self.batch_sz] = np.argmax(Yish_train_done, axis=1)

                test_error = error_rate(test_predictions, Ytest)
                train_error = error_rate(train_predictions, Ytrain)

                train_errors.append(train_error)
                train_costs.append(train_cost)

                test_errors.append(test_error)
                test_costs.append(test_cost)

                print('Epoch:', i, 'Train accuracy:', 1 - train_error, 'Train cost:', train_cost,
                     'Test accuracy', 1 - test_error, 'Test cost', test_cost)
                        
        return {'train costs': train_costs, 'train errors': train_errors, 
                'test costs': test_costs, 'test errors': test_errors}
    
    
    def pure_fit(self, Xtrain, Ytrain, Xtest, Ytest, optimizer=None, epochs=1, test_period=1):
        assert(optimizer is not None)
        assert(self.session is not None)
        Xtrain = Xtrain.astype(np.float32)
        Ytrain = Ytrain.argmax(axis=1)
        Xtest = Xtest.astype(np.float32)
        Ytest = Ytest.argmax(axis=1)
        
        # For training
        Yish = self.forward_train(self.X)
        cost = (tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Yish, labels=self.sparse_out) ), Yish)
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
            for j in tqdm(range(n_batches)):
                Xbatch = Xtrain[j*self.batch_sz:(j+1)*self.batch_sz]
                Ybatch = Ytrain[j*self.batch_sz:(j+1)*self.batch_sz]

                (train_cost_batch, y_ish), _ = self.session.run(train_op, feed_dict={self.X: Xbatch, self.sparse_out: Ybatch})
                # Use exponential decay for calculating loss and error
                train_cost = 0.99*train_cost + 0.01*train_cost_batch
                train_error_batch = error_rate(np.argmax(y_ish, axis=1), Ybatch)
                train_error = 0.99*train_error + 0.01*train_error_batch

            # Validating the network on test and part of train data
            if i % test_period == 0:
                # For test data
                test_cost = np.float32(0)
                test_predictions = np.zeros(len(Xtest))
                
                for k in range(len(Xtest) // self.batch_sz):
                    # Test data
                    Xtestbatch = Xtest[k*self.batch_sz:(k+1)*self.batch_sz]
                    Ytestbatch = Ytest[k*self.batch_sz:(k+1)*self.batch_sz]
                    Yish_test_done = self.session.run(Yish_test, feed_dict={self.X: Xtestbatch}) + EPSILON
                    test_cost += sparse_cross_entropy(Yish_test_done, Ytestbatch)
                    test_predictions[k*self.batch_sz:(k+1)*self.batch_sz] = np.argmax(Yish_test_done, axis=1)
                    
                # Collect and print data
                test_cost = test_cost / (len(Xtest) // self.batch_sz)
                test_error = error_rate(test_predictions, Ytest)
                test_errors.append(test_error)
                test_costs.append(test_cost)
                
                train_costs.append(train_cost)
                train_errors.append(train_error)
                

                print('Epoch:', i, 'Train accuracy:', 1 - train_error, 'Train cost:', train_cost,
                     'Test accuracy', 1 - test_error, 'Test cost', test_cost)
                        
        return {'train costs': train_costs, 'train errors': train_errors, 
                'test costs': test_costs, 'test errors': test_errors}
            
    