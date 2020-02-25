Convolutional layers
====================

ConvLayer
~~~~~~~~~
**Parameters**
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

------------------------------------------------------------------------------------------------------------------------

UpConvLayer
~~~~~~~~~~~
**Parameters**
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

.. important:: Shape is different from normal convolution since it's required by
    transposed convolution. Output feature maps go before input ones.

------------------------------------------------------------------------------------------------------------------------

DepthWiseConvLayer
~~~~~~~~~~~~~~~~~~
**Parameters**
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

------------------------------------------------------------------------------------------------------------------------

SeparableConvLayer
~~~~~~~~~~~~~~~~~~
**Parameters**
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

------------------------------------------------------------------------------------------------------------------------

AtrousConvLayer
~~~~~~~~~~~~~~~
**Parameters**
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

------------------------------------------------------------------------------------------------------------------------

