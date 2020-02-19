Layers
======


ConvLayer
^^^^^^^^^
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
