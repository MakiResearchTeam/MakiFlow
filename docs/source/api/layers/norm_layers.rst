Normalization layers
====================

BatchNormLayer
~~~~~~~~~~~~~~

Batch Normalization Procedure:
    X_normed = (X - mean) / variance
    X_final = X*gamma + beta
gamma and beta are defined by the NN, e.g. they are trainable.

**Parameters**
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
    mean : float
        Batch mean value. Used for initialization mean with pretrained value.
    var : float
        Batch variance value. Used for initialization variance with pretrained value.
    gamma : float
        Batchnorm gamma value. Used for initialization gamma with pretrained value.
    beta : float
        Batchnorm beta value. Used for initialization beta with pretrained value.

------------------------------------------------------------------------------------------------------------------------

GroupNormLayer
~~~~~~~~~~~~~~

GroupNormLayer Procedure:
    X_normed = (X - mean) / variance
    X_final = X*gamma + beta
There X (as original) have shape [N, H, W, C], but in this operation it will be [N, H, W, G, C // G].
GroupNormLayer normalized input on N and C // G axis.
gamma and beta are learned using gradient descent.

**Parameters**
    D : int
        Number of tensors to be normalized.
    decay : float
        Decay (momentum) for the moving mean and the moving variance.
    eps : float
        A small float number to avoid dividing by 0.
    G : int
        The number of groups that normalized. NOTICE! The number D must be divisible by G without remainder
    use_gamma : bool
        Use gamma in batchnorm or not.
    use_beta : bool
        Use beta in batchnorm or not.
    name : str
        Name of this layer.
    mean : float
        Batch mean value. Used for initialization mean with pretrained value.
    var : float
        Batch variance value. Used for initialization variance with pretrained value.
    gamma : float
        Batchnorm gamma value. Used for initialization gamma with pretrained value.
    beta : float

------------------------------------------------------------------------------------------------------------------------

NormalizationLayer
~~~~~~~~~~~~~~~~~~

NormalizationLayer Procedure:
    X_normed = (X - mean) / variance
    X_final = X*gamma + beta
There X have shape [N, H, W, C]. NormalizationLayer normqlized input on N axis
gamma and beta are learned using gradient descent.

**Parameters**
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
    mean : float
        Batch mean value. Used for initialization mean with pretrained value.
    var : float
        Batch variance value. Used for initialization variance with pretrained value.
    gamma : float
        Batchnorm gamma value. Used for initialization gamma with pretrained value.
    beta : float
        Batchnorm beta value. Used for initialization beta with pretrained value.

------------------------------------------------------------------------------------------------------------------------

InstanceNormLayer
~~~~~~~~~~~~~~~~~

InstanceNormLayer Procedure:
    X_normed = (X - mean) / variance
    X_final = X*gamma + beta

There X have shape [N, H, W, C]. InstanceNormLayer normalized input on N and C axis
gamma and beta are learned using gradient descent.

**Parameters**
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
    mean : float
        Batch mean value. Used for initialization mean with pretrained value.
    var : float
        Batch variance value. Used for initialization variance with pretrained value.
    gamma : float
        Batchnorm gamma value. Used for initialization gamma with pretrained value.
    beta : float
        Batchnorm beta value. Used for initialization beta with pretrained value.