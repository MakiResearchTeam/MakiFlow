Other layers
------------

BiasLayer
~~~~~~~~~

BiasLayer adds a bias vector of dimension D to a tensor.

**Parameters**
    D : int
        Dimension of bias vector.
    name : str
        Name of this layer.

------------------------------------------------------------------------------------------------------------------------

DenseLayer
~~~~~~~~~~
**Parameters**
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

------------------------------------------------------------------------------------------------------------------------

ScaleLayer
~~~~~~~~~~

ScaleLayer is used to multiply input MakiTensor on `init_value`, which is trainable variable.

**Parameters**
    init_value : int
        The initial value which need to multiply by input.
    name : str
        Name of this layer.

------------------------------------------------------------------------------------------------------------------------