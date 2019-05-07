from __future__ import absolute_import

# For loading model architecture
import json

from makiflow.advanced_layers import *
from makiflow.conv_model import ConvModel
from makiflow.layers import *
from makiflow.save_recover.activation_converter import ActivationConverter
from makiflow.ssd.detector_classifier import DetectorClassifier
from makiflow.ssd.detector_classifier_block import DetectorClassifierBlock
from makiflow.ssd.ssd_model import SSDModel


class Builder:
    def convmodel_from_json(json_path):
        """Creates and returns ConvModel from json.json file contains its architecture"""
        json_file = open(json_path)
        json_value = json_file.read()
        architecture_dict = json.loads(json_value)
        name = architecture_dict['name']
        input_shape = architecture_dict['input_shape']
        output_shape = architecture_dict['output_shape']
        
        layers = []
        for layer_dict in architecture_dict['layers']:
            layers.append(Builder.__layer_from_dict(layer_dict))
        
        print('Model is recovered.')
        
        return ConvModel(layers=layers, input_shape=input_shape, output_shape=output_shape, name=name)
    
    
    def ssd_from_json(json_path):
        """Creates and returns SSDModel from json.json file contains its architecture"""
        json_file = open(json_path)
        json_value = json_file.read()
        architecture_dict = json.loads(json_value)
        name = architecture_dict['name']
        input_shape = architecture_dict['input_shape']
        num_classes = architecture_dict['num_classes']
        
        dc_blocks = []
        for dc_block_dict in architecture_dict['dc_blocks']:
            dc_blocks.append(Builder.__dc_block_from_dict(dc_block_dict))
            
        print('Model is recovered.')
            
        return SSDModel(dc_blocks=dc_blocks, input_shape=input_shape, num_classes=num_classes, name=name)
    
    
    def __dc_block_from_dict(dc_block_dict):
        """Creates and returns DetectorClassifierBlock from dictionary"""
        # Create layers for the dc_block
        layers = []
        for layer_dict in dc_block_dict['layers']:
            layers.append(Builder.__layer_from_dict(layer_dict))
        
        # Create detector classifier for dc_block
        detector_classifier = Builder.__detector_classifier_from_dict(dc_block_dict['detector_classifier'])
        return DetectorClassifierBlock(layers=layers, detector_classifier=detector_classifier)
    
    
    def __detector_classifier_from_dict(dc_dict):
        """Creates and returns DetectorClassifier from dictionary"""
        params = dc_dict['params']
        
        name = params['name']
        class_number = params['class_number']
        dboxes = params['dboxes']
        kw = params['classifier_shape'][0]
        kh = params['classifier_shape'][1]
        in_f = params['classifier_shape'][2]
        
        return DetectorClassifier(kw=kw, kh=kh, in_f=in_f, class_number=class_number, dboxes=dboxes, name=name)
        
        
        
            
            
    def __layer_from_dict(layer_dict):
        """Creates and returns Layer from dictionary"""
        params = layer_dict['params']
        uni_dict = {
            'ConvLayer': Builder.__conv_layer_from_dict,
            'DenseLayer': Builder.__dense_layer_from_dict,
            'BatchNormLayer': Builder.__batchnorm_layer_from_dict,
            'MaxPoolLayer': Builder.__maxpool_layer_from_dict,
            'AvgPoolLayer': Builder.__avgpool_layer_from_dict,
            'FlattenLayer': Builder.__flatten_layer_from_dict,
            'DropoutLayer': Builder.__dropout_layer_from_dict,
            'ActivationLayer': Builder.__activation_layer_from_dict,
            'ResnetIndentityBlock': Builder.__resnet_convblock_from_dict
        }
        return uni_dict[layer_dict['type']](params)
    
    
    def __flatten_layer_from_dict(params):
        return FlattenLayer()
    
    
    def __resnet_convblock_from_dict(params):
        name = params['name']
        in_f1 = params['in_f1']
        out_f1 = params['out_f1']
        out_f2 = params['out_f2']
        return ResnetIndentityBlock(in_f1=in_f1, out_f1=out_f1, out_f2=out_f2, name=name)
        
    
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
        