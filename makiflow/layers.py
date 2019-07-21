from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from makiflow.save_recover.activation_converter import ActivationConverter


class Layer(object):
    def __init__(self):
        self.params = []
        self.named_params_dict = {}

    def forward(self, X, is_training=False):
        pass

    def get_params(self):
        return self.params

    def get_params_dict(self):
        """
        This data is used for correct saving and loading models using TensorFlow checkpoint files.
        """
        return self.named_params_dict

    def to_dict(self):
        """
        This data is used for converting the model's architecture to json.json file.
        """
        pass

    def __call__(self, *args, **kwargs):
        pass


class ConvLayer(Layer):
    def __init__(self, kw, kh, in_f, out_f, name, stride=1, padding='SAME', activation=tf.nn.relu,
                 W=None, b=None):
        """
        :param kw - kernel width.
        :param kh - kernel height.
        :param in_f - number of input feature maps. Treat as color channels if this layer
            is first one.
        :param out_f - number of output feature maps (number of filters).
        :param padding - padding mode for convolution operation.
        :param activation - activation function. Set None if you don't need activation.
        :param W - filter's weights. This value is used for the filter initialization with pretrained filters.
        :param b - bias' weights. This value is used for the bias initialization with pretrained bias.
        """
        Layer.__init__(self)
        self.shape = (kw, kh, in_f, out_f)
        self.stride = stride
        self.padding = padding
        self.f = activation

        self.name = str(name)
        self.name_conv = 'ConvKernel{}x{}_in{}_out{}_id_'.format(kw, kh, in_f, out_f) + str(name)
        self.name_bias = 'ConvBias{}x{}_in{}_out{}_id_'.format(kw, kh, in_f, out_f) + str(name)

        if W is None:
            W = np.random.randn(*self.shape) * np.sqrt(2.0 / np.prod(self.shape[:-1]))
        if b is None:
            b = np.zeros(out_f)

        self.W = tf.Variable(W.astype(np.float32), name=self.name_conv)
        self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
        self.params = [self.W, self.b]
        self.named_params_dict = {self.name_conv: self.W, self.name_bias: self.b}

    def forward(self, X, is_training=False):
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, self.stride, self.stride, 1], padding=self.padding)
        conv_out = tf.nn.bias_add(conv_out, self.b)
        if self.f is None:
            return conv_out
        return self.f(conv_out)

    def copy_from_keras_layers(self, layer):
        W, b = layer.get_weights()
        op1 = self.W.assign(W)
        op2 = self.b.assign(b)

        self.session.run((op1, op2))

    def to_dict(self):
        return {
            'type': 'ConvLayer',
            'params': {
                'name': self.name,
                'shape': list(self.shape),
                'stride': self.stride,
                'padding': self.padding,
                'activation': ActivationConverter.activation_to_str(self.f)
            }
        }


class DenseLayer(Layer):
    def __init__(self, input_shape, output_shape, name, activation=tf.nn.relu, init_type='xavier',
                 W=None, b=None):
        """
        :param input_shape - number represents input shape. Example: 500.
        :param output_shape - number represents output shape. You can treat it as number of neurons. Example: 100.
        :param activation - activation function. Set None if you don't need activation.
        :param init_type - name of the weights initialization way: `xavier` or `lasange`. For relu like activations
            `xavier` initialization performs better.
        :param W - matrix weights. Used for initialisation dense weights with pretrained weights.
        :param b - bias weights. Used for initialisation dense bias with pretrained bias.
        """
        Layer.__init__(self)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.f = activation

        if W is None:
            W = np.random.randn(input_shape, output_shape)
            # Perform Xavier initialization
            if init_type == 'xavier':
                W /= (input_shape + output_shape) / 2
            # Perform Lasange initialization
            else:
                W *= np.sqrt(12 / (input_shape + output_shape))

        if b is None:
            b = np.zeros(output_shape)

        self.name = str(name)
        self.name_dense = 'DenseMat{}x{}_id_'.format(input_shape, output_shape) + str(name)
        self.name_bias = 'DenseBias{}x{}_id_'.format(input_shape, output_shape) + str(name)

        self.W = tf.Variable(W.astype(np.float32), name=self.name_dense)
        self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
        self.params = [self.W, self.b]
        self.named_params_dict = {self.name_dense: self.W, self.name_bias: self.b}

    def forward(self, X, is_training=False):
        out = tf.matmul(X, self.W) + self.b
        if self.f is None:
            return out
        return self.f(out)

    def copy_from_keras_layers(self, layer):
        W, b = layer.get_weights()
        op1 = self.W.assign(W)
        op2 = self.b.assign(b)
        self.session.run((op1, op2))

    def to_dict(self):
        return {
            'type': 'DenseLayer',
            'params': {
                'name': self.name,
                'input_shape': self.input_shape,
                'output_shape': self.output_shape,
                'activation': ActivationConverter.activation_to_str(self.f)
            }
        }


class BatchNormLayer(Layer):
    def __init__(self, D, name,
                 mean=None, var=None, gamma=None, beta=None):
        """
        :param D - number of tensors to be normalized.
        :param mean - batch mean value. Used for initialization mean with pretrained value.
        :param var - batch variance value. Used for initialization variance with pretrained value.
        :param gamma - batchnorm gamma value. Used for initialization gamma with pretrained value.
        :param beta - batchnorm beta value. Used for initialization beta with pretrained value.
        Batch Noramlization Procedure:
            X_normed = (X - mean) / variance
            X_final = X*gamma + beta
        gamma and beta are defined by the NN, e.g. they are trainable.
        """
        Layer.__init__(self)
        self.D = D

        if mean is None:
            mean = np.zeros(D)
        if var is None:
            var = np.ones(D)

        # These variables are needed to change the mean and variance of the batch after
        # the batchnormalization: result*gamma + beta
        # beta - offset
        # gamma - scale
        if beta is None:
            beta = np.zeros(D)
        if gamma is None:
            gamma = np.ones(D)

        self.name = str(name)
        self.name_mean = 'BatchMean{}_id_'.format(D) + str(name)
        self.name_var = 'BatchVar{}_id_'.format(D) + str(name)
        self.name_gamma = 'BatchGamma{}_id_'.format(D) + str(name)
        self.name_beta = 'BatchBeta{}_id_'.format(D) + str(name)

        self.running_mean = tf.Variable(mean.astype(np.float32), trainable=False, name=self.name_mean)
        self.running_variance = tf.Variable(var.astype(np.float32), trainable=False, name=self.name_var)
        self.gamma = tf.Variable(gamma.astype(np.float32), name=self.name_gamma)
        self.beta = tf.Variable(beta.astype(np.float32), name=self.name_beta)

        self.params = [self.running_mean, self.running_variance, self.gamma, self.beta]
        self.named_params_dict = {self.name_mean: self.running_mean, self.name_var: self.running_variance,
                                  self.name_gamma: self.gamma, self.name_beta: self.beta}

    def forward(self, X, is_training=False, decay=0.9):
        """
        :param decay - this argument is responsible for how fast batchnorm layer is trained. Values between 0.9 and 0.999 are
        commonly used.
        """
        # These if statements check if we do batchnorm for convolution or dense
        if len(X.shape) == 4:
            # conv
            axes = [0, 1, 2]
        else:
            # dense
            axes = [0]
        if is_training:
            batch_mean, batch_var = tf.nn.moments(X, axes=axes)
            update_running_mean = tf.assign(
                self.running_mean,
                self.running_mean * decay + batch_mean * (1 - decay)
            )
            update_running_variance = tf.assign(
                self.running_variance,
                self.running_variance * decay + batch_var * (1 - decay)
            )
            with tf.control_dependencies([update_running_mean, update_running_variance]):
                out = tf.nn.batch_normalization(
                    X,
                    batch_mean,
                    batch_var,
                    self.beta,
                    self.gamma,
                    1e-4
                )
        else:
            out = tf.nn.batch_normalization(
                X,
                self.running_mean,
                self.running_variance,
                self.beta,
                self.gamma,
                1e-4
            )
        return out

    def copy_from_keras_layer(self, layer):
        # only 1 layer to copy from
        # order:
        # gamma, beta, moving mean, moving variance
        gamma, beta, running_mean, running_variance = layer.get_weights()
        op1 = self.running_mean.assign(running_mean)
        op2 = self.running_variance.assign(running_variance)
        op3 = self.gamma.assign(gamma)
        op4 = self.beta.assign(beta)
        self.session.run((op1, op2, op3, op4))

    def to_dict(self):
        return {
            'type': 'BatchNormLayer',
            'params': {
                'name': self.name,
                'D': self.D,
            }
        }


class MaxPoolLayer(Layer):
    def __init__(self, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
        Layer.__init__(self)
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def forward(self, X, is_training=False):
        return tf.nn.max_pool(
            X,
            ksize=self.ksize,
            strides=self.strides,
            padding=self.padding
        )

    def to_dict(self):
        return {
            'type': 'MaxPoolLayer',
            'params': {
                'ksize': self.ksize,
                'strides': self.strides,
                'padding': self.padding
            }
        }


class AvgPoolLayer(Layer):
    def __init__(self, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
        Layer.__init__(self)
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def forward(self, X, is_training=False):
        return tf.nn.avg_pool(
            X,
            ksize=self.ksize,
            strides=self.strides,
            padding=self.padding
        )

    def to_dict(self):
        return {
            'type': 'AvgPoolLayer',
            'params': {
                'ksize': self.ksize,
                'strides': self.strides,
                'padding': self.padding
            }
        }


class ActivationLayer(Layer):
    def __init__(self, activation=tf.nn.relu):
        Layer.__init__(self)
        if activation is None:
            raise WrongInput("Activation can't None")
        self.f = activation

    def forward(self, X, is_training=False):
        return self.f(X)

    def to_dict(self):
        return {
            'type': 'ActivationLayer',
            'params': {
                'activation': ActivationConverter.activation_to_str(self.f)
            }
        }


class FlattenLayer(Layer):
    def __init__(self):
        Layer.__init__(self)
        pass

    def forward(self, X, is_training=False):
        return tf.contrib.layers.flatten(X)

    def to_dict(self):
        return {
            'type': 'FlattenLayer',
            'params': {}
        }


class DropoutLayer(Layer):
    def __init__(self, p_keep=0.9):
        Layer.__init__(self)
        self.p_keep = p_keep

    def forward(self, X, is_training=False):
        return tf.nn.dropout(X, self.p_keep)

    def to_dict(self):
        return {
            'type': 'DropoutLayer',
            'params': {
                'p_keep': self.p_keep
            }
        }
