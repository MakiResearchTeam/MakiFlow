from __future__ import absolute_import
from makiflow.layers import Layer, ConvLayer, BatchNormLayer, ActivationLayer


# Reference: https://arxiv.org/pdf/1603.05027v2.pdf

# TIPS:
# 1) DON'T USE ACTIVATION FUNCTION BEFORE THE BLOCK
# 2) DON'T USE BATCHNORMALIZATION BEFORE THE BLOCK


class IdentityBlock(Layer):
    def __init__(self, main_branch, name):
        """
        Parameters
        ----------
        main_branch : list
            List of different layers performed on the main branch. 
            WARNING! OUTPUT TENSOR OF MAIN BRANCH MUST BE THE SAME SHAPE AS THE SKIP BRANCH!
        """
        Layer.__init__(self)
        self.name = str(name)
        self.main_branch = main_branch
        # Collect trainable params
        self.params = []
        for layer in self.main_branch:
            self.params += layer.get_params()
        # Get params and store them into python dictionary in order to save and load them correctly later
        self.named_params_dict = {}
        for layer in self.main_branch:
            self.named_params_dict.update(layer.get_params_dict())

    def get_params(self):
        return self.params

    def get_params_dict(self):
        return self.named_params_dict

        
    def forward(self, X, is_training=False):
        FX = X
        for layer in self.main_branch:
            FX = layer.forward(FX, is_training)
            
        return FX + X
        
        
    def to_dict(self):
        desc =  {
            'type':'IdentityBlock',
            'params': {
                'name': self.name,
                'main_branch': []
            }
        }

        for layer in self.main_branch:
            desc['params']['main_branch'].append(layer.to_dict())
        
        return desc
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
        