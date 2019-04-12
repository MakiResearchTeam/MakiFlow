from __future__ import absolute_import
from makiflow.layers import *
from makiflow.conv_model import ConvModel
from makiflow.save_recover.activation_converter import ActivationConverter
# For loading model architecture
import json

class Builder:
    def convmodel_from_json(json_path):
        json_file = open(json_path)
        json_value = json_file.read()
        architecture_dict = json.loads(json_value)
        name = architecture_dict['name']
        input_shape = architecture_dict['input_shape']
        output_shape = architecture_dict['output_shape']
        
        layers = []
        for layer_dict in architecture_dict['layers']:
            layers.append(Builder.__layer_from_dict(layer_dict))
        
        return ConvModel(layers=layers, input_shape=input_shape, output_shape=output_shape, name=name)
        
            
    def __layer_from_dict(layer_dict):
        params = layer_dict['params']
        uni_dict = {
            'ConvLayer': Builder.__conv_layer_from_dict,
            'DenseLayer': Builder.__dense_layer_from_dict,
            'BatchNormLayer': Builder.__batchnorm_layer_from_dict,
            'MaxPoolLayer': Builder.__maxpool_layer_from_dict,
            'AvgPoolLayer': Builder.__avgpool_layer_from_dict,
            'FlattenLayer': Builder.__flatten_layer_from_dict,
            'DropoutLayer': Builder.__dropout_layer_from_dict,
            'Activation': Builder.__activation_layer_from_dict
        }
        return uni_dict[layer_dict['type']](params)
    
    
    def __flatten_layer_from_dict(params):
        return FlattenLayer()
        
    
    def __conv_layer_from_dict(params):
        name = params['name']
        kw = params['shape'][0]
        kh = params['shape'][1]
        in_f = params['shape'][2]
        out_f = params['shape'][3]
        stride = params['stride']
        padding = params['padding']
        activation = ActivationConverter.str_to_activation(params['activation'])
        return ConvLayer(kw=kw, kh=kh, in_f=in_f, out_f=out_f, 
                         stride=stride, name=name, padding=padding, activation=activation)
    
    
    def __dense_layer_from_dict(params):
        name = params['name']
        input_shape = params['input_shape']
        output_shape = params['output_shape']
        activation = ActivationConverter.str_to_activation(params['activation'])
        return DenseLayer(input_shape=input_shape, output_shape=output_shape, activation=activation, name=name)
    
    
    def __batchnorm_layer_from_dict(params):
        name = params['name']
        D = params['D']
        return BatchNormLayer(D=D, name=name)
    
    
    def __maxpool_layer_from_dict(params):
        ksize = params['ksize']
        strides = params['strides']
        padding = params['padding']
        return MaxPoolLayer(ksize=ksize, strides=strides, padding=padding)
    
    
    def __avgpool_layer_from_dict(params):
        ksize = params['ksize']
        strides = params['strides']
        padding = params['padding']
        return MaxPoolLayer(ksize=ksize, strides=strides, padding=padding)
    
    
    def __activation_layer_from_dict(params):
        activation = ActivationConverter.str_to_activation(params['activation'])
        return ActivationLayer(activation=activation)
    
    
    def __dropout_layer_from_dict(params):
        p_keep = params['p_keep']
        return DropoutLayer(p_keep=p_keep)
        