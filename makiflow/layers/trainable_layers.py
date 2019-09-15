from __future__ import absolute_import
import numpy as np
import tensorflow as tf

from makiflow.layers.activation_converter import ActivationConverter
from makiflow.layers.sf_layer import SimpleForwardLayer


class ConvLayer(SimpleForwardLayer):
    def __init__(self, kw, kh, in_f, out_f, name, stride=1, padding='SAME', activation=tf.nn.relu,
                 kernel_initializer='he', use_bias=True, W=None, b=None):
        """
        Parameters
        ----------
        kw : int
            Kernel width.
        kh : int
            Kernel height.
        in_f : int
            Number of input feature maps. Treat as color channels if this layer
            is first one.
        out_f : int 
            Number of output feature maps (number of filters).
        stride : int
            Defines the stride of the convolution.
        padding : str
            Padding mode for convolution operation. Options: 'SAME', 'VALID' (case sensitive). 
        activation : tensorflow function
            Activation function. Set None if you don't need activation.
        W : numpy array
            Filter's weights. This value is used for the filter initialization with pretrained filters.
        b : numpy array
            Bias' weights. This value is used for the bias initialization with pretrained bias.
        use_bias : bool
            Add bias to the output tensor.
        name : str
            Name of this layer.
        """
        self.shape = (kw, kh, in_f, out_f)
        self.stride = stride
        self.padding = padding
        self.f = activation
        self.use_bias = use_bias
        self.init_type = kernel_initializer

        name = str(name)

        if W is None:
            W = init_conv_kernel(kw, kh, in_f, out_f, kernel_initializer)
        if b is None:
            b = np.zeros(out_f)

        self.name_conv = 'ConvKernel_{}x{}_in{}_out{}_id_'.format(kw, kh, in_f, out_f) + name
        self.W = tf.Variable(W.astype(np.float32), name=self.name_conv)
        params = [self.W]
        named_params_dict = {self.name_conv: self.W}
        if use_bias:
            self.name_bias = 'ConvBias_{}x{}_in{}_out{}_id_'.format(kw, kh, in_f, out_f) + name
            self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
            params += [self.b]
            named_params_dict[self.name_bias] = self.b

        super().__init__(name, params, named_params_dict)

    def _forward(self, X):
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, self.stride, self.stride, 1], padding=self.padding)
        if self.use_bias:
            conv_out = tf.nn.bias_add(conv_out, self.b)
        if self.f is None:
            return conv_out
        return self.f(conv_out)

    def _training_forward(self, X):
        return self._forward(X)

    def to_dict(self):
        return {
            'type': 'ConvLayer',
            'params': {
                'name': self._name,
                'shape': list(self.shape),
                'stride': self.stride,
                'padding': self.padding,
                'activation': ActivationConverter.activation_to_str(self.f),
                'use_bias': self.use_bias,
                'init_type': self.init_type
            }

        }


class UpConvLayer(SimpleForwardLayer):
    def __init__(self, kw, kh, in_f, out_f, name, size=(2, 2), padding='SAME', activation=tf.nn.relu,
                 kernel_initializer='he', use_bias=True, W=None, b=None):
        """
        Parameters
        ----------
        kw : int
            Kernel width.
        kh : int
            Kernel height.
        in_f : int
            Number of input feature maps. Treat as color channels if this layer
            is first one.
        out_f : int 
            Number of output feature maps (number of filters).
        size : tuple
            Tuple of two ints - factors of the size of the output feature map.
            Example: feature map with spatial dimension (n, m) will produce
            output feature map of size (a*n, b*m) after performing up-convolution
            with `size` (a, b).
        padding : str
            Padding mode for convolution operation. Options: 'SAME', 'VALID' (case sensitive). 
        activation : tensorflow function
            Activation function. Set None if you don't need activation.
        W : numpy array
            Filter's weights. This value is used for the filter initialization with pretrained filters.
        b : numpy array
            Bias' weights. This value is used for the bias initialization with pretrained bias.
        use_bias : bool
            Add bias to the output tensor.
        """
        # Shape is different from normal convolution since it's required by 
        # transposed convolution. Output feature maps go before input ones.
        self.shape = (kw, kh, out_f, in_f)
        self.size = size
        self.strides = [1, *size, 1]
        self.padding = padding
        self.f = activation
        self.use_bias = use_bias
        self.init_type = kernel_initializer

        name = str(name)

        if W is None:
            W = init_conv_kernel(kw, kh, out_f, in_f, kernel_initializer)
        if b is None:
            b = np.zeros(out_f)

        self.name_conv = 'UpConvKernel_{}x{}_out{}_in{}_id_'.format(kw, kh, out_f, in_f) + name
        self.W = tf.Variable(W.astype(np.float32), name=self.name_conv)
        params = [self.W]
        named_params_dict = {self.name_conv: self.W}
        if use_bias:
            self.name_bias = 'UpConvBias_{}x{}_in{}_out{}_id_'.format(kw, kh, in_f, out_f) + name
            self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
            params += [self.b]
            named_params_dict[self.name_bias] = self.b

        super().__init__(name, params, named_params_dict)

    def _forward(self, X):
        out_shape = X.get_shape().as_list()
        out_shape[1] *= self.size[0]
        out_shape[2] *= self.size[1]
        # out_f
        out_shape[3] = self.shape[2]
        conv_out = tf.nn.conv2d_transpose(
            X, self.W,
            output_shape=out_shape, strides=self.strides, padding=self.padding
        )
        if self.use_bias:
            conv_out = tf.nn.bias_add(conv_out, self.b)

        if self.f is None:
            return conv_out
        return self.f(conv_out)

    def _training_forward(self, X):
        return self._forward(X)

    def to_dict(self):
        return {
            'type': 'UpConvLayer',
            'params': {
                'name': self._name,
                'shape': list(self.shape),
                'size': self.size,
                'padding': self.padding,
                'activation': ActivationConverter.activation_to_str(self.f),
                'use_bias': self.use_bias,
                'init_type': self.init_type
            }
        }

class BiasLayer(SimpleForwardLayer):
    def __init__(self, D, name):
        """
        BiasLayer adds a bias vector of dimension D to a tensor.

        Parameters
        ----------
        D : int
            Dimension of bias vector.
        name : str
            Name of this layer.
        """
        self.D = D
        self.name = name

        b = np.zeros(D)
        params = []
        self.bias_name = f'BiasLayer_{D}' + name
        self.b = tf.Variable(b.astype(np.float32), name=self.bias_name)
        params = [self.b]
        named_params_dict = {self.bias_name: self.b}

        super().__init__(name, params, named_params_dict)

    def _forward(self, X):    
        return tf.nn.bias_add(X, self.b)

    def _training_forward(self, X):
        return self._forward(X)

    def to_dict(self):
        return {
            'type': 'BiasLayer',
            'params': {
                'name': self._name,
                'D': self.D,
            }
        }

class DepthWiseConvLayer(SimpleForwardLayer):
    def __init__(self, kw, kh, in_f, multiplier, name, stride=1, padding='SAME', rate=[1,1],
                 kernel_initializer='he', use_bias=True, activation=tf.nn.relu, W=None, b=None):
        """
        Parameters
        ----------
        kw : int
            Kernel width.
        kh : int
            Kernel height.
        in_f : int
            Number of input feature maps. Treat as color channels if this layer
            is first one.
        multiplier : int 
            Number of output feature maps equals `in_f`*`multiplier`.
        stride : int
            Defines the stride of the convolution.
        padding : str
            Padding mode for convolution operation. Options: 'SAME', 'VALID' (case sensitive). 
        activation : tensorflow function
            Activation function. Set None if you don't need activation.
        W : numpy array
            Filter's weights. This value is used for the filter initialization with pretrained filters.
        use_bias : bool
            Add bias to the output tensor. 
        name : str
            Name of this layer.
        """
        assert (len(rate) == 2)
        self.shape = (kw, kh, in_f, multiplier)
        self.stride = stride
        self.padding = padding
        self.f = activation
        self.use_bias = use_bias
        self.rate = rate
        self.init_type = kernel_initializer

        name = str(name)

        if W is None:
            W = init_conv_kernel(kw, kh, in_f, multiplier, kernel_initializer)
        if b is None:
            b = np.zeros(in_f * multiplier)

        self.name_conv = 'DepthWiseConvKernel_{}x{}_in{}_out{}_id_'.format(kw, kh, in_f, multiplier) + name
        self.W = tf.Variable(W.astype(np.float32), name=self.name_conv)
        params = [self.W]
        named_params_dict = {self.name_conv: self.W}
        if use_bias:
            self.bias_name = f'DepthWiseConvBias_{in_f * multiplier}' + name
            self.b = tf.Variable(b.astype(np.float32), name=self.bias_name)
            params += [self.b]
            named_params_dict[self.bias_name] = self.b

        super().__init__(name, params, named_params_dict)

    def _forward(self, X):
        conv_out = tf.nn.depthwise_conv2d(
            input=X,
            filter=self.W,
            strides=[1, self.stride, self.stride, 1],
            padding=self.padding,
            rate=self.rate,
        )
        if self.use_bias:
            conv_out = tf.nn.bias_add(conv_out, self.b)
        if self.f is None:
            return conv_out
        return self.f(conv_out)

    def _training_forward(self, X):
        return self._forward(X)

    def to_dict(self):
        return {
            'type': 'DepthWiseLayer',
            'params': {
                'name': self._name,
                'shape': list(self.shape),
                'stride': self.stride,
                'padding': self.padding,
                'activation': ActivationConverter.activation_to_str(self.f),
                'use_bias': self.use_bias,
                'init_type': self.init_type,
                'rate': self.rate,
            }
        }


class SeparableConvLayer(SimpleForwardLayer):
    def __init__(self, kw, kh, in_f, out_f, multiplier, name, stride=1, padding='SAME',
                 dw_kernel_initializer='xavier_gaussian_inf', pw_kernel_initializer='he',
                 use_bias=True, activation=tf.nn.relu,
                 W_dw=None, W_pw=None, b=None):
        """
        Parameters
        ----------
        kw : int
            Kernel width.
        kh : int
            Kernel height.
        in_f : int
            Number of the input feature maps. Treat as color channels if this layer
            is first one.
        out_f : int
            Number of the output feature maps after pointwise convolution,
            i.e. it is depth of the final output tensor.
        multiplier : int 
            Number of output feature maps after depthwise convolution equals `in_f`*`multiplier`.
        stride : int
            Defines the stride of the convolution.
        padding : str
            Padding mode for convolution operation. Options: 'SAME', 'VALID' (case sensitive). 
        activation : tensorflow function
            Activation function. Set None if you don't need activation.
        W_dw : numpy array
            Filter's weights. This value is used for the filter initialization.
        use_bias : bool
            Add bias to the output tensor.
        name : str
            Name of this layer.  
        """
        self.dw_shape = (kw, kh, in_f, multiplier)
        self.out_f = out_f
        self.stride = stride
        self.padding = padding
        self.f = activation
        self.use_bias = use_bias
        self.dw_init_type = dw_kernel_initializer
        self.pw_init_type = pw_kernel_initializer

        name = str(name)

        if W_dw is None:
            W_dw = init_conv_kernel(kw, kh, in_f, multiplier, dw_kernel_initializer)
        if W_pw is None:
            W_pw = init_conv_kernel(1, 1, multiplier * in_f, out_f, pw_kernel_initializer)
        if b is None:
            b = np.zeros(out_f)

        self.name_DW = f'DWConvKernel_{kw}x{kh}_in{in_f}_out{multiplier}_id_{name}'
        self.name_PW = f'PWConvKernel_{1}x{1}_in{in_f * multiplier}_out{out_f}_id_{name}'
        self.W_dw = tf.Variable(W_dw, name=self.name_DW)
        self.W_pw = tf.Variable(W_pw, name=self.name_PW)
        params = [self.W_dw, self.W_pw]
        named_params_dict = {
            self.name_DW: self.W_dw,
            self.name_PW: self.W_pw,
        }
        if use_bias:
            self.bias_name = f'SeparableConvBias_{out_f}' + name
            self.b = tf.Variable(b.astype(np.float32), name=self.bias_name)
            params += [self.b]
            named_params_dict[self.bias_name] = self.b

        super().__init__(name, params, named_params_dict)

    def _forward(self, X):
        conv_out = tf.nn.separable_conv2d(
            input=X,
            depthwise_filter=self.W_dw,
            pointwise_filter=self.W_pw,
            strides=[1, self.stride, self.stride, 1],
            padding=self.padding,
        )
        if self.use_bias:
            conv_out = tf.nn.bias_add(conv_out, self.b)
        if self.f is None:
            return conv_out
        return self.f(conv_out)

    def _training_forward(self, X):
        return self._forward(X)

    def to_dict(self):
        return {
            'type': 'SeparableConvLayer',
            'params': {
                'name': self._name,
                'dw_shape': list(self.dw_shape),
                'out_f': self.out_f,
                'stride': self.stride,
                'padding': self.padding,
                'activation': ActivationConverter.activation_to_str(self.f),
                'use_bias': self.use_bias,
                'dw_init_type': self.dw_init_type,
                'pw_init_type': self.pw_init_type
            }
        }


class DenseLayer(SimpleForwardLayer):
    def __init__(self, in_d, out_d, name, activation=tf.nn.relu,
                 mat_initializer='he', use_bias=True, W=None, b=None):
        """
        Paremeters
        ----------
        in_d : int 
            Dimensionality of the input vector. Example: 500.
        out_d : int 
            Dimensionality of the output vector. Example: 100.
        activation : TensorFlow function
            Activation function. Set to None if you don't need activation.
        W : numpy ndarray
            Used for initialization the weight matrix.
        b : numpy ndarray
            Used for initialisation the bias vector.
        use_bias : bool
            Add bias to the output tensor.
        name : str
            Name of this layer.
        """

        self.input_shape = in_d
        self.output_shape = out_d
        self.f = activation
        self.use_bias = use_bias
        self.init_type = mat_initializer

        if W is None:
            W = init_dense_mat(in_d, out_d, mat_initializer)

        if b is None:
            b = np.zeros(out_d)

        name = str(name)
        self.name_dense = 'DenseMat_{}x{}_id_'.format(in_d, out_d) + name
        self.W = tf.Variable(W, name=self.name_dense)
        params = [self.W]
        named_params_dict = {self.name_dense: self.W}
        if use_bias:
            self.name_bias = 'DenseBias_{}x{}_id_'.format(in_d, out_d) + name
            self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
            params += [self.b]
            named_params_dict[self.name_bias] = self.b

        super().__init__(name, params, named_params_dict)

    def _forward(self, X):
        out = tf.matmul(X, self.W)
        if self.use_bias:
            out = out + self.b
        if self.f is None:
            return out
        return self.f(out)

    def _training_forward(self, X):
        return self._forward(X)

    def to_dict(self):
        return {
            'type': 'DenseLayer',
            'params': {
                'name': self._name,
                'input_shape': self.input_shape,
                'output_shape': self.output_shape,
                'activation': ActivationConverter.activation_to_str(self.f),
                'use_bias': self.use_bias,
                'init_type': self.init_type
            }
        }


class AtrousConvLayer(SimpleForwardLayer):   
    def __init__(self, kw, kh, in_f, out_f, rate, name, padding='SAME', activation=tf.nn.relu,
                 kernel_initializer='he', use_bias=True, W=None, b=None):
        """
        Parameters
        ----------
        kw : int
            Kernel width.
        kh : int
            Kernel height.
        in_f : int
            Number of input feature maps. Treat as color channels if this layer
            is first one.
        out_f : int 
            Number of output feature maps (number of filters).
        rate : int
            A positive int. The stride with which we sample input values across the height and width dimensions
        stride : int
            Defines the stride of the convolution.
        padding : str
            Padding mode for convolution operation. Options: 'SAME', 'VALID' (case sensitive). 
        activation : tensorflow function
            Activation function. Set None if you don't need activation.
        W : numpy array
            Filter's weights. This value is used for the filter initialization with pretrained filters.
        b : numpy array
            Bias' weights. This value is used for the bias initialization with pretrained bias.
        use_bias : bool
            Add bias to the output tensor.
        name : str
            Name of this layer.
        """
        self.shape = (kw, kh, in_f, out_f)
        self.rate = rate
        self.padding = padding
        self.f = activation
        self.use_bias = use_bias
        self.init_type = kernel_initializer

        name = str(name)
        self.name_conv = f'AtrousConvKernel_{kw}x{kh}_in{in_f}_out{out_f}_rate{rate}_id_{name}'

        if W is None:
            W = init_conv_kernel(kw, kh, in_f, out_f, kernel_initializer)
        if b is None:
            b = np.zeros(out_f)

        self.W = tf.Variable(W.astype(np.float32), name=self.name_conv)
        params = [self.W]
        named_params_dict = {self.name_conv: self.W}

        if use_bias:
            self.name_bias = f'AtrousConvBias_{kw}x{kh}_in{in_f}_out{out_f}_rate{rate}_id_{name}'
            self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
            params += [self.b]
            named_params_dict[self.name_bias] = self.b

        super().__init__(name, params, named_params_dict)

    def _forward(self, X):
        conv_out = tf.nn.atrous_conv2d(X, self.W, self.rate, self.padding)
        if self.use_bias:
            conv_out = tf.nn.bias_add(conv_out, self.b)
        if self.f is None:
            return conv_out
        return self.f(conv_out)

    def _training_forward(self, x):
        return self._forward(x)

    def to_dict(self):
        return {
            'type': 'AtrousConvLayer',
            'params': {
                'name': self._name,
                'shape': list(self.shape),
                'rate': self.rate,
                'padding': self.padding,
                'activation': ActivationConverter.activation_to_str(self.f),
                'use_bias': self.use_bias,
                'init_type': self.init_type
            }
        }


class BatchNormLayer(SimpleForwardLayer):
    def __init__(self, D, name, decay=0.9, eps=1e-4, use_gamma=True,
                    use_beta=True, mean=None, var=None, gamma=None, beta=None):
        """
        :param mean - batch mean value. Used for initialization mean with pretrained value.
        :param var - batch variance value. Used for initialization variance with pretrained value.
        :param gamma - batchnorm gamma value. Used for initialization gamma with pretrained value.
        :param beta - batchnorm beta value. Used for initialization beta with pretrained value.
        Batch Noramlization Procedure:
            X_normed = (X - mean) / variance
            X_final = X*gamma + beta
        gamma and beta are defined by the NN, e.g. they are trainable.

        Parameters
        ----------
        D : int
            Number of tensors to be normalized.
        decay : float
            Decay (momentum) for the moving mean and the moving variance.
        eps : float
            A small float number to avoid dividing by 0.
        use_gamma : bool
            Use gamma in batchnorm or not.
        use_beta : bool
            Use beta in batchnorm or not.
        name : str
            Name of this layer. 
        """
        self.D = D
        self.decay = decay
        self.eps = eps
        self.use_gamma = use_gamma
        self.use_beta = use_beta

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

        name = str(name)
        self.name_mean = 'BatchMean_{}_id_'.format(D) + name
        self.name_var = 'BatchVar_{}_id_'.format(D) + name

        self.running_mean = tf.Variable(mean.astype(np.float32), trainable=False, name=self.name_mean)
        self.running_variance = tf.Variable(var.astype(np.float32), trainable=False, name=self.name_var)

        params = []
        named_params_dict = {self.name_mean: self.running_mean, self.name_var: self.running_variance}
        
        # Create gamma
        self.name_gamma = 'BatchGamma_{}_id_'.format(D) + name
        self.gamma = tf.Variable(gamma.astype(np.float32), name=self.name_gamma)
        named_params_dict[self.name_gamma] = self.gamma

        if use_gamma:
            params += [self.gamma]
        
        # Create beta
        self.name_beta = 'BatchBeta_{}_id_'.format(D) + name
        self.beta = tf.Variable(beta.astype(np.float32), name=self.name_beta)
        named_params_dict[self.name_beta] = self.beta

        if use_beta: 
            params += [self.beta]   

        super().__init__(name, params, named_params_dict)

    def _forward(self, X):
        return tf.nn.batch_normalization(
            X,
            self.running_mean,
            self.running_variance,
            self.beta,
            self.gamma,
            self.eps
        )

    def _training_forward(self, X):
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

        batch_mean, batch_var = tf.nn.moments(X, axes=axes)
        update_running_mean = tf.assign(
            self.running_mean,
            self.running_mean * self.decay + batch_mean * (1 - self.decay)
        )
        update_running_variance = tf.assign(
            self.running_variance,
            self.running_variance * self.decay + batch_var * (1 - self.decay)
        )
        with tf.control_dependencies([update_running_mean, update_running_variance]):
            out = tf.nn.batch_normalization(
                X,
                batch_mean,
                batch_var,
                self.beta,
                self.gamma,
                self.eps
            )

        return out

    def to_dict(self):
        return {
            'type': 'BatchNormLayer',
            'params': {
                'name': self._name,
                'D': self.D,
                'decay': self.decay,
                'eps': self.eps,
                'use_beta': self.use_beta,
                'use_gamma': self.use_gamma,
            }
        }

# Some initializate methods
# Initializations define the way to set the initial random weights of MakiFlow layers.
def init_conv_kernel(kw, kh, in_f, out_f, kernel_initializer):
    W = np.random.randn(kw, kh, in_f, out_f)
    if kernel_initializer == 'xavier_gaussian_avg':
        W *= np.sqrt(3. / (kw * kh * in_f + kw * kh * out_f))

    elif kernel_initializer == 'xavier_gaussian_inf':
        W *= np.sqrt(1. / (kw * kh * in_f))

    elif kernel_initializer == 'xavier_uniform_avg':
        W = np.random.uniform(low=-1., high=1.0, size=[kw, kh, in_f, out_f])
        W *= np.sqrt(6. / (kw * kh * in_f + kw * kh * out_f))

    elif kernel_initializer == 'xavier_uniform_inf':
        W = np.random.uniform(low=-1., high=1.0, size=[kw, kh, in_f, out_f])
        W *= np.sqrt(3. / (kw * kh * in_f))

    elif kernel_initializer == 'he':
        W *= np.sqrt(2. / (kw * kh * in_f))

    elif kernel_initializer == 'lasange':
        W = np.random.uniform(low=-1., high=1.0, size=[kw, kh, in_f, out_f])
        W *= np.sqrt(12. / (kw * kh * in_f + kw * kh * out_f))

    return W.astype(np.float32)


def init_dense_mat(in_d, out_d, mat_initializer):
    W = np.random.randn(in_d, out_d)
    if mat_initializer == 'xavier_gaussian':
        W *= np.sqrt(3. / (in_d + out_d))

    elif mat_initializer == 'xavier_uniform':
        W = np.random.uniform(low=-1., high=1.0, size=[in_d, out_d])
        W *= np.sqrt(6. / (in_d + out_d))

    elif mat_initializer == 'he':
        W *= np.sqrt(2. / (in_d))

    elif mat_initializer == 'lasange':
        W = np.random.uniform(low=-1., high=1.0, size=[in_d, out_d])
        W *= np.sqrt(12. / (in_d + out_d))

    return W.astype(np.float32)
