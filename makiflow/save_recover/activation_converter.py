from tensorflow.nn import relu, sigmoid, tanh, softmax

class ActivationConverter:
    """ This class is used for saving models. We need to know what activation
    function is used in a particular layer, so we map them with some string values 
    like tensorflow.nn.relu -> relu. This way we can then recover the model cause we
    know what activation function is used.
    """
    
    def activation_to_str(activation):
        return {
            relu: 'relu',
            sigmoid: 'sigmoid',
            tanh: 'tanh',
            softmax: 'softmax',
            None: 'None'
        }[activation]
    
    
    def str_to_activation(activation_name):
        return {
            'relu': relu,
            'sigmoid': sigmoid,
            'tanh': tanh,
            'softmax': softmax,
            'None': None
        }[activation_name]