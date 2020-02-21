Tensor manipulation layers
==========================

ReshapeLayer
------------

ReshapeLayer is used to changes size from some input_shape to new_shape (include batch_size and color dimension).

**Parameters**
    new_shape : list
        Shape of output object.
    name : str
        Name of this layer.

------------------------------------------------------------------------------------------------------------------------

MulByAlphaLayer
---------------

MulByAlphaLayer is used to multiply input MakiTensor by `alpha`.

**Parameters**
    alpha : int
        The constant to multiply by.
    name : str
        Name of this layer.

------------------------------------------------------------------------------------------------------------------------

SumLayer
--------

SumLayer is used add input MakiTensors together.

**Parameters**
    name : str
        Name of this layer.

------------------------------------------------------------------------------------------------------------------------

ConcatLayer
-----------

Concatenates input MakiTensors along certain `axis`.

**Parameters**
    axis : int
        Dimension along which to concatenate.
    name : str
        Name of this layer.

------------------------------------------------------------------------------------------------------------------------

ZeroPaddingLayer
----------------

Adds rows and columns of zeros at the top, bottom, left and right side of an image tensor.

**Parameters**
    padding : list
        List the number of additional rows and columns in the appropriate directions.
        For example like [ [top,bottom], [left,right] ]
   name : str
        Name of this layer.

------------------------------------------------------------------------------------------------------------------------

GlobalMaxPoolLayer
------------------

Performs global maxpooling.
NOTICE! The output tensor will be flattened, i.e. will have a shape of [batch size, num features].
