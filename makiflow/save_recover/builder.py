from __future__ import absolute_import

# For loading model architecture
import json

from makiflow.advanced_layers import *
from makiflow.conv_model import ConvModel
from makiflow.layers import *
from makiflow.rnn_layers import *
from makiflow.save_recover.activation_converter import ActivationConverter
from makiflow.ssd.detector_classifier import DetectorClassifier
from makiflow.ssd.detector_classifier_block import DetectorClassifierBlock
from makiflow.ssd.ssd_model import SSDModel
from makiflow.rnn_models.text_recognizer import TextRecognizer


class Builder:
    @staticmethod
    def convmodel_from_json(json_path, batch_size=None):
        """Creates and returns ConvModel from json.json file contains its architecture"""
        json_file = open(json_path)
        json_value = json_file.read()
        architecture_dict = json.loads(json_value)
        name = architecture_dict['name']
        input_shape = architecture_dict['input_shape']
        if batch_size is not None:
            input_shape[0] = batch_size
        num_classes = architecture_dict['num_classes']
        
        layers = []
        for layer_dict in architecture_dict['layers']:
            layers.append(Builder.__layer_from_dict(layer_dict))
        
        print('Model is recovered.')
        
        return ConvModel(layers=layers, input_shape=input_shape, num_classes=num_classes, name=name)
    
    @staticmethod
    def ssd_from_json(json_path, batch_size=None):
        """Creates and returns SSDModel from json.json file contains its architecture"""
        json_file = open(json_path)
        json_value = json_file.read()
        architecture_dict = json.loads(json_value)
        name = architecture_dict['name']
        input_shape = architecture_dict['input_shape']
        if batch_size is not None:
            input_shape[0] = batch_size
        num_classes = architecture_dict['num_classes']
        
        dc_blocks = []
        for dc_block_dict in architecture_dict['dc_blocks']:
            dc_blocks.append(Builder.__dc_block_from_dict(dc_block_dict))
            
        print('Model is recovered.')
            
        return SSDModel(dc_blocks=dc_blocks, input_shape=input_shape, num_classes=num_classes, name=name)

    @staticmethod
    def text_recognizer_from_json(json_path, batch_size=None):
        """Creates and returns TextRecognizer from json.json file contains its architecture"""
        json_file = open(json_path)
        json_value = json_file.read()
        architecture_dict = json.loads(json_value)
        name = architecture_dict['name']
        input_shape = architecture_dict['input_shape']
        if batch_size is not None:
            input_shape[0] = batch_size
        chars = architecture_dict['chars']
        max_seq_length = architecture_dict['max_seq_length']
        decoder_type = architecture_dict['decoder_type']

        cnn_layers = []
        for layer in architecture_dict['cnn_layers']:
            cnn_layers.append(Builder.__layer_from_dict(layer))
        rnn_layers = []
        for layer in architecture_dict['rnn_layers']:
            rnn_layers.append(Builder.__layer_from_dict(layer))

        if batch_size is not None:
            input_shape = [batch_size, *input_shape[1:]]

        return TextRecognizer(
            cnn_layers=cnn_layers,
            rnn_layers=rnn_layers,
            input_shape=input_shape,
            chars=chars,
            max_seq_length=max_seq_length,
            decoder_type=decoder_type,
            name=name
        )
    
    @staticmethod
    def __dc_block_from_dict(dc_block_dict):
        """Creates and returns DetectorClassifierBlock from dictionary"""
        # Create layers for the dc_block
        layers = []
        for layer_dict in dc_block_dict['layers']:
            layers.append(Builder.__layer_from_dict(layer_dict))
        
        # Create detector classifier for dc_block
        detector_classifier = Builder.__detector_classifier_from_dict(dc_block_dict['detector_classifier'])
        return DetectorClassifierBlock(layers=layers, detector_classifier=detector_classifier)
    
    @staticmethod
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
        
        
        
            
    @staticmethod       
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
            'ResnetIndentityBlock': Builder.__resnet_convblock_from_dict,
            'IdentityBlock': Builder.__identityblock_from_dict,
            'GRULayer': Builder.__gru_layer_from_dict,
            'LSTMLayer': Builder.__lstm_layer_from_dict,
            'RNNBlock': Builder.__rnnblock_from_dict,
            'StemBlock':Builder.__stem_from_dict,
            'Inception_resnet_A_Block':Builder.__inception_resnet_A_from_dict,
            'Reduction_A_block':Builder.__reduction_A_from_dict,
            'Inception_resnet_B_block':Builder.__inception_resnet_B_from_dict,
            'Reduction_B_block':Builder.__reduction_B_from_dict,
        }
        return uni_dict[layer_dict['type']](params)
    
    @staticmethod
    def __flatten_layer_from_dict(params):
        return FlattenLayer()
    
    @staticmethod
    def __resnet_convblock_from_dict(params):
        name = params['name']
        in_f1 = params['in_f1']
        out_f1 = params['out_f1']
        return ResnetIndentityBlock(in_f1=in_f1, out_f1=out_f1, name=name)
    
    @staticmethod
    def __identityblock_from_dict(params):
        name = params['name']
        main_branch = []
        for layer in params['main_branch']:
            main_branch.append(Builder.__layer_from_dict(layer))
        return IdentityBlock(
            main_branch=main_branch,
            name=name
            )
        
    @staticmethod
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
    
    @staticmethod
    def __dense_layer_from_dict(params):
        name = params['name']
        input_shape = params['input_shape']
        output_shape = params['output_shape']
        activation = ActivationConverter.str_to_activation(params['activation'])
        return DenseLayer(input_shape=input_shape, output_shape=output_shape, activation=activation, name=name)
    
    @staticmethod
    def __batchnorm_layer_from_dict(params):
        name = params['name']
        D = params['D']
        return BatchNormLayer(D=D, name=name)
    
    @staticmethod
    def __maxpool_layer_from_dict(params):
        ksize = params['ksize']
        strides = params['strides']
        padding = params['padding']
        return MaxPoolLayer(ksize=ksize, strides=strides, padding=padding)
    
    @staticmethod
    def __avgpool_layer_from_dict(params):
        ksize = params['ksize']
        strides = params['strides']
        padding = params['padding']
        return MaxPoolLayer(ksize=ksize, strides=strides, padding=padding)
    
    @staticmethod
    def __activation_layer_from_dict(params):
        activation = ActivationConverter.str_to_activation(params['activation'])
        return ActivationLayer(activation=activation)
    
    @staticmethod
    def __dropout_layer_from_dict(params):
        p_keep = params['p_keep']
        return DropoutLayer(p_keep=p_keep)

    @staticmethod
    def __gru_layer_from_dict(params):
        num_cells = params['num_cells']
        input_dim = params['input_dim']
        seq_length = params['seq_length']
        name = params['name']
        dynamic = params['dynamic']
        bidirectional = params['bidirectional']
        activation = ActivationConverter.str_to_activation(params['activation'])
        return GRULayer(
            num_cells=num_cells,
            input_dim=input_dim,
            seq_length=seq_length,
            name=name,
            activation=activation,
            dynamic=dynamic,
            bidirectional=bidirectional
        )


    @staticmethod
    def __lstm_layer_from_dict(params):
        num_cells = params['num_cells']
        input_dim = params['input_dim']
        seq_length = params['seq_length']
        name = params['name']
        dynamic = params['dynamic']
        bidirectional = params['bidirectional']
        activation = ActivationConverter.str_to_activation(params['activation'])
        return LSTMLayer(
            num_cells=num_cells,
            input_dim=input_dim,
            seq_length=seq_length,
            name=name,
            activation=activation,
            dynamic=dynamic,
            bidirectional=bidirectional
        )


    @staticmethod
    def __rnnblock_from_dict(params):
        seq_length = params['seq_length']
        dynamic = params['dynamic']
        bidirectional = params['bidirectional']
        rnn_layers = Builder.__layer_from_dict(params['rnn_layers'])
        return RNNBlock(
            rnn_layers=rnn_layers,
            seq_length=seq_length,
            dynamic=dynamic,
            bidirectional=bidirectional
        )
    

    @staticmethod
    def __embedding_layer_from_dict(params):
        num_embeddings = params['num_embeddings']
        dim = params['dim']
        name = params['name']
        return EmbeddingLayer(
            num_embeddings=num_embeddings,
            dim=dim,
            name=name
        )

    @staticmethod
    def __stem_from_dict(params):
        name = params['name']
        in_f = params['in_f']
        out_f = params['out_f']
        activation =  ActivationConverter.str_to_activation(params['activation'])
        return StemBlock(in_f=in_f, out_f=out_f,activation=activation, name=name)

    @staticmethod
    def __inception_resnet_A_from_dict(params):
        name = params['name']
        in_f = params['in_f']
        out_f = params['out_f']
        activation =  ActivationConverter.str_to_activation(params['activation'])
        return Inception_A(in_f=in_f, out_f=out_f,activation=activation, name=name)  
    
    @staticmethod
    def __reduction_A_from_dict(params):
        name = params['name']
        in_f = params['in_f']
        out_f = params['out_f']
        activation =  ActivationConverter.str_to_activation(params['activation'])
        return Reduction_A(in_f=in_f, out_f=out_f,activation=activation, name=name)  
    
    @staticmethod
    def __inception_resnet_B_from_dict(params):
        name = params['name']
        in_f = params['in_f']
        out_f = params['out_f']
        activation =  ActivationConverter.str_to_activation(params['activation'])
        return Inception_B(in_f=in_f, out_f=out_f,activation=activation, name=name)

    @staticmethod
    def __reduction_B_from_dict(params):
        name = params['name']
        in_f = params['in_f']
        out_f = params['out_f']
        activation =  ActivationConverter.str_to_activation(params['activation'])
        return Reduction_B(in_f=in_f, out_f=out_f,activation=activation, name=name)  
      
    

        