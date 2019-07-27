from __future__ import absolute_import
from makiflow.layers import MakiLayer, ConvLayer, BatchNormLayer, ActivationLayer


# Reference: https://arxiv.org/pdf/1603.05027v2.pdf

# TIPS:
# 1) DON'T USE ACTIVATION FUNCTION BEFORE THE BLOCK
# 2) DON'T USE BATCHNORMALIZATION BEFORE THE BLOCK


class ResnetIndentityBlock(MakiLayer):
    def __init__(self, in_f1, out_f1, name, pretrained_weights=None):
        """
        Parameters
        ----------
        in_f1 : int
            Number of the input feature maps relative to the first convolution in the block.
        out_f1 : int
            Number of the output feature maps of the first convolution in the block.
        pretrained_weights : list
            List contains pretrained weights for all convolutional and batchnorm layers in the block
            arranged in the following order: 
                [conv_W1, conv_b1, conv_W2, conv_b2,
                bn_mean1, bn_var1, bn_gamma1, bn_beta1,
                bn_mean2, bn_var2, bn_gamma2, bn_beta2]
        
        Notes
        -----
            Data flow scheme: 
            batchnorm -> activation -> convolution -> batchnorm -> activation -> convolution
            At the end of the block input and last convolution output are added together:
            final_output = input + last_conv_output.

        Tips
        ----
            Don't use batchnormalization before the block.
        """
        MakiLayer.__init__(self)
        self.in_f1 = in_f1
        self.out_f1 = out_f1
        if pretrained_weights is not None:
            assert(len(pretrained_weights) == 12)
            self.__init_with_pretrained_weighst(pretrained_weights)
        else:
            self.name = 'ResnetIndentityBlock_'+str(name)
            self.batchnorm1 = BatchNormLayer(in_f1, name=self.name+'__batchnorm1')
            self.activation1 = ActivationLayer()
            self.convolution1 = ConvLayer(3, 3, in_f1, out_f1, name=self.name+'__convolution1', activation=None)

            self.batchnorm2 = BatchNormLayer(out_f1, name=self.name+'__batchnorm2')
            self.activation2 = ActivationLayer()
            self.convolution2 = ConvLayer(3, 3, out_f1, in_f1, name=self.name+'__convolution2', activation=None)
            
        self.layers = [self.batchnorm1, self.activation1, self.convolution1, 
                       self.batchnorm2, self.activation2, self.convolution2]
        # Collect trainable params
        self.params = []
        for layer in self.layers:
            self.params += layer.get_params()
        # Get params and store them into python dictionary in order to save and load them correctly later
        self.named_params_dict = {}
        for layer in self.layers:
            self.named_params_dict.update(layer.get_params_dict())

        
    def __init_with_pretrained_weighst(self, ptw):
        # FIRST PART
        # First batch normalization
        bn_mean1 = ptw[4]
        bn_var1 = ptw[5]
        bn_gamma1 = ptw[6]
        bn_beta1 = ptw[7]
        self.batchnorm1 = BatchNormLayer(self.in_f1, name=self.name+'__batchnorm1',
                                        mean=bn_mean1, var=bn_var1, gamma=bn_gamma1, beta=bn_beta1)
        # First activation
        self.activation1 = ActivationLayer()  # ReLU
        # First convolution
        conv_W1 = ptw[0]
        conv_b1 = ptw[1]
        self.convolution1 = ConvLayer(3, 3, self.in_f1, self.out_f1, name=self.name+'__convolution1', activation=None,
                                     W=conv_W1, b=conv_b1)
        
        # SECOND PART
        # Second batchnormalization
        bn_mean2 = ptw[8]
        bn_var2 = ptw[9]
        bn_gamma2 = ptw[10]
        bn_beta2 = ptw[11]
        self.batchnorm2 = BatchNormLayer(self.out_f1, name=self.name+'__batchnorm2',
                                        mean=bn_mean2, var=bn_var2, gamma=bn_gamma2, beta=bn_beta2)
        # First activation
        self.activation2 = ActivationLayer()  # ReLU
        # Second convolution
        conv_W2 = ptw[2]
        conv_b2 = ptw[3]
        self.convolution2 = ConvLayer(3, 3, self.out_f1, self.in_f1, name=self.name+'__convolution2', activation=None,
                                     W=conv_W2, b=conv_b2)
        
        
    def forward(self, X, is_training=False):
        FX = X
        for layer in self.layers:
            FX = layer.forward(FX, is_training)
            
        return FX + X
        
        
    def to_dict(self):
        return {
            'type':'ResnetIndentityBlock',
            'params': {
                'name': self.name,
                'in_f1': self.in_f1,
                'out_f1': self.out_f1,
            }
        }
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
        