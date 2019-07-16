import tensorflow as tf
import json
import numpy as np


class Encoder:
    def __init__(self, input_shape, layers, name):
        """
        Parameters
        ----------
        layers : list
            List of layers model consist of.
            Example:
            [
                DenseLayer(300),
                DenseLayer(100),
            ]
        input_shape : list
            List of ints represent input shape: [batch_size, input_size].
        name : str
            Name of the model.
        """
        
        self.name = name
        self.input_shape = input_shape
        self.batch_sz = input_shape[0]
        self.layers = layers

        # Collect all the params for later initialization
        self.params = []
        for layer in self.layers:
            self.params += layer.get_params()


        # Get params and store them into python dictionary in order to save and load them correctly later
        self.named_params_dict = {}
        for layer in self.layers:
            self.named_params_dict.update(layer.get_params_dict())


        self.X = tf.placeholder(tf.float32, shape=input_shape)
        self.out = self.forward(self.X,is_training=False)

    
    def get_params(self):
        """
        Returns
        -------
            list
                Contains all the model possibly trainable parameters.
        """
        return self.params

    def get_named_params(self):
        """
        Returns
        -------
            list
                Contains all the model possibly trainable parameters paired with 
                their names. This is used for saving the autoencoder.
        """
        return self.named_params_dict

    
    def set_session(self, session):
        self.session = session
        init_op = tf.variables_initializer(self.params)
        self.session.run(init_op)


    def forward(self, X,is_training=True):
        for layer in self.layers:
            X = layer.forward(X,is_training)
        return X

    def encode(self, X):
        assert(self.session is not None)
        n_batches = X.shape[0] // self.batch_sz
        result = []
        for i in range(n_batches):
            Xbatch = X[i*self.batch_sz:(i+1)*self.batch_sz]
            result.append(
                self.session.run(
                    self.out,
                    feed_dict={self.X: Xbatch}
                )
            )
        return np.vstack(result)

    
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
            'input_shape': self.input_shape
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