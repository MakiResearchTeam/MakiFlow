from __future__ import absolute_import

# For loading model architecture
import json

from makiflow.models.classificator import Classificator
from makiflow.layers import *
from makiflow.rnn_layers import *
from makiflow.save_recover.activation_converter import ActivationConverter
from makiflow.models.ssd import DetectorClassifier
from makiflow.models.ssd import SSDModel


class Builder:

    @staticmethod
    def restore_graph(outputs,graph_info):
        used = {}
        coll_tensors = {}

        def restore_in_and_out_x(from_):
            # from_ - name of layer
            parent_layer_info = graph_info[from_]
            if used.get(from_) is None:
                used[from_] = True
                # like "to"
                all_parent_names = parent_layer_info['parent_tensor_names']
                # store ready tensors
                takes = [] 
                answer = None
                if len(all_parent_names) != 0:
                    # All layer except input layer
                    layer = Builder.__layer_from_dict(parent_layer_info['parent_layer_info'])
                    for elem in all_parent_names:
                        takes += [restore_in_and_out_x(elem)]
                    answer = layer(takes[0] if len(takes) == 1 else takes)
                else:
                    # Input layer
                    temp = {}
                    temp.update({'type':parent_layer_info['type'],'params':parent_layer_info['params']})
                    answer = Builder.__layer_from_dict(temp)

                coll_tensors[from_] = answer
                return answer
            else:
                return coll_tensors[from_]

        for name_output in outputs:
            restore_in_and_out_x(name_output)

        return coll_tensors

    @staticmethod
    def classificator_from_json(json_path, batch_size=None):
        """Creates and returns ConvModel from json.json file contains its architecture"""
        json_file = open(json_path)
        json_value = json_file.read()
        json_info = json.loads(json_value)

        output_tensor_name = json_info['model_info']['output_mt']
        input_tensor_name = json_info['model_info']['input_mt']
        model_name = json_info['model_info']['name']

        MakiTensors_of_model = json_info['graph_info']

        # dict {NameTensor : Info about this tensor}
        graph_info = {}

        for tensor in MakiTensors_of_model:
            graph_info[tensor['name']] = tensor

        del MakiTensors_of_model
        
        inputs_outputs = Builder.restore_graph([output_tensor_name],graph_info)
        out_x = inputs_outputs[output_tensor_name]
        in_x = inputs_outputs[input_tensor_name]
        print('Model is restored!')
        return Classificator(input=in_x,output=out_x,model_name=model_name)
    
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
        
        return DetectorClassifier(kw=kw, kh=kh, in_f=in_f, num_classes=class_number, dboxes=dboxes, name=name)
         
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
            'GRULayer': Builder.__gru_layer_from_dict,
            'LSTMLayer': Builder.__lstm_layer_from_dict,
            'RNNBlock': Builder.__rnnblock_from_dict,
            'InputLayer':Builder.__input_layer_from_dict,
            'SumLayer':Builder.__sum_layer_from_dict,
            'ConcatLayer' : Builder.__concat_layer_from_dict,
            'MultiOnAlphaLayer' : Builder.__MultiOnAlphaLayer_layer_from_dict,
            'ZeroPaddingLayer' : Builder.__zeropadding_layer_from_dict,
            'GlobalMaxPoolLayer' : Builder.__globalmaxpoollayer_from_dict,
            'GlobalAvgPoolLayer' : Builder.__globalavgpoollayer_from_dict,
            'UpSamplingLayer' : Builder.__upsampling_layer_from_dict,
            'UpConvLayer' : Builder.__upconv_layer_from_dict,
            'ReshapeLayer' : Builder.__reshape_layer_from_dict,
            'DepthWiseLayer' : Builder.__depthwise_layer_from_dict,
        }
        return uni_dict[layer_dict['type']](params)

    @staticmethod
    def __depthwise_layer_from_dict(params):
        name = params['name']
        kw = params['shape'][0]
        kh = params['shape'][1]
        in_f = params['shape'][2]
        multiplier = params['shape'][3]
        padding = params['padding']
        stride = params['stride']
        activation = ActivationConverter.str_to_activation(params['activation'])
        return DepthWiseLayer(kw=kw, kh=kh, in_f=in_f, multiplier=multiplier, padding=padding,
                            stride=stride, activation=activation, name=name)

    @staticmethod
    def __reshape_layer_from_dict(params):
        name = params['name']
        new_shape = params['new_shape']
        return ReshapeLayer(new_shape=new_shape, name=name)

    @staticmethod
    def __upconv_layer_from_dict(params):
        name = params['name']
        kw = params['shape'][0]
        kh = params['shape'][1]
        in_f = params['shape'][2]
        out_f = params['shape'][3]
        size = params['size']
        padding = params['padding']
        activation = ActivationConverter.str_to_activation(params['activation'])
        return UpConvLayer(kw=kw, kh=kh, in_f=in_f, out_f=out_f, 
                         size=size, name=name, padding=padding, activation=activation)

    @staticmethod
    def __upsampling_layer_from_dict(params):
        name = params['name']
        size = params['size']
        return UpSamplingLayer(name=name, size=size)

    @staticmethod
    def __globalmaxpoollayer_from_dict(params):
        name = params['name']
        return GlobalMaxPoolLayer(name=name)
    
    @staticmethod
    def __globalavgpoollayer_from_dict(params):
        name = params['name']
        return GlobalAvgPoolLayer(name=name)

    @staticmethod
    def __zeropadding_layer_from_dict(params):
        name = params['name']
        padding = params['padding']
        return ZeroPaddingLayer(padding=padding, name=name)

    @staticmethod
    def __MultiOnAlphaLayer_layer_from_dict(params):
        name = params['name']
        alpha = params['alpha']
        return LamdaLayer(alpha=alpha, name=name)

    @staticmethod
    def __concat_layer_from_dict(params):
        name = params['name']
        axis = params['axis']
        return ConcatLayer(name=name, axis=axis)

    @staticmethod
    def __sum_layer_from_dict(params):
        name = params['name']
        return SumLayer(name=name)

    @staticmethod
    def __flatten_layer_from_dict(params):
        name = params['name']
        return FlattenLayer(name=name)

    @staticmethod
    def __input_layer_from_dict(params):
        input_shape = params['input_shape']
        name = params['name']
        return InputLayer(name=name, input_shape = input_shape)    
        
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
        return DenseLayer(in_d=input_shape, out_d=output_shape, activation=activation, name=name)
    
    @staticmethod
    def __batchnorm_layer_from_dict(params):
        name = params['name']
        D = params['D']
        return BatchNormLayer(D=D, name=name)
    
    @staticmethod
    def __maxpool_layer_from_dict(params):
        return MaxPoolLayer(**params)
    
    @staticmethod
    def __avgpool_layer_from_dict(params):
        ksize = params['ksize']
        strides = params['strides']
        padding = params['padding']
        return AvgPoolLayer(ksize=ksize, strides=strides, padding=padding)
    
    @staticmethod
    def __activation_layer_from_dict(params):
        name = params['name']
        activation = ActivationConverter.str_to_activation(params['activation'])
        return ActivationLayer(name=name,activation=activation)
    
    @staticmethod
    def __dropout_layer_from_dict(params):
        name = params['name']
        p_keep = params['p_keep']
        return DropoutLayer(name=name,p_keep=p_keep)

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