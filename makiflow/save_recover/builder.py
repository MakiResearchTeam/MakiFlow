from __future__ import absolute_import

# For loading model architecture
import json

from makiflow.models.classificator import Classificator
from makiflow.layers import *
from makiflow.layers.activation_converter import ActivationConverter
from makiflow.models import DetectorClassifier
from makiflow.models import SSDModel
from makiflow.models import Segmentator
from makiflow.models import TextRecognizer


class Builder:

    @staticmethod
    def classificator_from_json(json_path, batch_size=None):
        """Creates and returns ConvModel from json.json file contains its architecture"""
        json_file = open(json_path)
        json_value = json_file.read()
        json_info = json.loads(json_value)

        output_tensor_name = json_info['model_info']['output_mt']
        input_tensor_name = json_info['model_info']['input_mt']
        model_name = json_info['model_info']['name']

        graph_info = json_info['graph_info']

        inputs_outputs = Builder.restore_graph([output_tensor_name], graph_info, batch_size)
        out_x = inputs_outputs[output_tensor_name]
        in_x = inputs_outputs[input_tensor_name]
        print('Model is restored!')
        return Classificator(input=in_x, output=out_x, name=model_name)

    @staticmethod
    def ssd_from_json(json_path, batch_size=None):
        """Creates and returns SSDModel from json.json file contains its architecture"""
        json_file = open(json_path)
        json_value = json_file.read()
        architecture_dict = json.loads(json_value)
        name = architecture_dict['name']
        # Collect names of the MakiTensors that are inputs for the DetectorClassifiers
        # for restoring the graph.
        dcs_dicts = architecture_dict['dcs']
        outputs = []
        for dcs_dict in dcs_dicts:
            outputs += [dcs_dict['reg_x'], dcs_dict['class_x']]

        graph_info = architecture_dict['graph_info']
        inputs_outputs = Builder.restore_graph(outputs, graph_info, batch_size)
        # Restore all the DetectorClassifiers
        dcs = []
        for dc_dict in architecture_dict['dcs']:
            reg_x = inputs_outputs[dc_dict['reg_x']]
            class_x = inputs_outputs[dc_dict['class_x']]
            dcs.append(Builder.__detector_classifier_from_dict(dc_dict, reg_x, class_x))
        input_name = architecture_dict['input_s']
        input_s = inputs_outputs[input_name]

        print('Model is recovered.')

        return SSDModel(dcs=dcs, input_s=input_s, name=name)

    @staticmethod
    def __detector_classifier_from_dict(dc_dict, reg_x, class_x):
        """Creates and returns DetectorClassifier from dictionary"""
        params = dc_dict['params']
        name = params['name']
        class_number = params['class_number']
        dboxes = params['dboxes']

        rkw = params['rkw']
        rkh = params['rkh']
        rin_f = params['rin_f']

        ckw = params['ckw']
        ckh = params['ckh']
        cin_f = params['cin_f']

        return DetectorClassifier(
            reg_fms=reg_x, rkw=rkw, rkh=rkh, rin_f=rin_f,
            class_fms=reg_x, ckw=ckw, ckh=ckh, cin_f=cin_f,
            num_classes=class_number, dboxes=dboxes, name=name
        )

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
    def segmentator_from_json(json_path, batch_size=None):
        """Creates and returns ConvModel from json.json file contains its architecture"""
        json_file = open(json_path)
        json_value = json_file.read()
        json_info = json.loads(json_value)

        output_tensor_name = json_info['model_info']['output']
        input_tensor_name = json_info['model_info']['input_s']
        model_name = json_info['model_info']['name']

        MakiTensors_of_model = json_info['graph_info']

        inputs_outputs = Builder.restore_graph(
            [output_tensor_name], MakiTensors_of_model, batch_size
        )
        out_x = inputs_outputs[output_tensor_name]
        in_x = inputs_outputs[input_tensor_name]
        print('Model is restored!')
        return Segmentator(input_s=in_x, output=out_x, name=model_name)

    # -----------------------------------------------------------LAYERS RESTORATION----------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __layer_from_dict(layer_dict):
        """Creates and returns Layer from dictionary"""
        params = layer_dict['params']
        uni_dict = {
            'UpConvLayer': Builder.__upconv_layer_from_dict,
            'SeparableConvLayer': Builder.__separableconv_layer_from_dict,
            'AtrousConvLayer': Builder.__atrousconv_layer_from_dict,
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
            'InputLayer': Builder.__input_layer_from_dict,
            'SumLayer': Builder.__sum_layer_from_dict,
            'ConcatLayer': Builder.__concat_layer_from_dict,
            'MultiOnAlphaLayer': Builder.__mulbyalpha_layer_from_dict,
            'ZeroPaddingLayer': Builder.__zeropadding_layer_from_dict,
            'GlobalMaxPoolLayer': Builder.__globalmaxpoollayer_from_dict,
            'GlobalAvgPoolLayer': Builder.__globalavgpoollayer_from_dict,
            'DepthWiseLayer': Builder.__depthwise_layer_from_dict,
            'ReshapeLayer': Builder.__reshape_layer_from_dict,
            'UpSamplingLayer': Builder.__upsampling_layer_from_dict,
            'BiasLayer': Builder.__bias_layer_from_dict,
            'ResizeLayer': Builder.__resize_layer_from_dict,
        }
        return uni_dict[layer_dict['type']](params)

    @staticmethod
    def __resize_layer_from_dict(params):
        new_shape = params['new_shape']
        name = params['name']
        align_corners = params['align_corners']
        interpolation = params['interpolation']
        return ResizeLayer(interpolation=interpolation, new_shape=new_shape, name=name,
                           align_corners=align_corners)

    @staticmethod
    def __bias_layer_from_dict(params):
        name = params['name']
        D = params['D']
        return BiasLayer(D=D, name=name)

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
        return InputLayer(name=name, input_shape=input_shape)

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
        init_type = params['init_type']
        use_bias = params['use_bias']
        return ConvLayer(
            kw=kw, kh=kh, in_f=in_f, out_f=out_f,
            stride=stride, name=name, padding=padding, activation=activation,
            kernel_initializer=init_type, use_bias=use_bias
        )

    @staticmethod
    def __upconv_layer_from_dict(params):
        name = params['name']
        kw = params['shape'][0]
        kh = params['shape'][1]
        in_f = params['shape'][3]
        out_f = params['shape'][2]
        padding = params['padding']
        size = params['size']
        activation = ActivationConverter.str_to_activation(params['activation'])
        init_type = params['init_type']
        use_bias = params['use_bias']
        return UpConvLayer(
            kw=kw, kh=kh, in_f=in_f, out_f=out_f, size=size,
            name=name, padding=padding, activation=activation,
            kernel_initializer=init_type, use_bias=use_bias
        )

    @staticmethod
    def __depthwise_layer_from_dict(params):
        name = params['name']
        kw = params['shape'][0]
        kh = params['shape'][1]
        in_f = params['shape'][2]
        multiplier = params['shape'][3]
        padding = params['padding']
        stride = params['stride']
        init_type = params['init_type']
        use_bias = params['use_bias']
        rate = params['rate']
        activation = ActivationConverter.str_to_activation(params['activation'])
        return DepthWiseConvLayer(
            kw=kw, kh=kh, in_f=in_f, multiplier=multiplier, padding=padding,
            stride=stride, activation=activation, name=name, rate=rate,
            kernel_initializer=init_type, use_bias=use_bias,
        )

    @staticmethod
    def __separableconv_layer_from_dict(params):
        name = params['name']
        kw = params['dw_shape'][0]
        kh = params['dw_shape'][1]
        in_f = params['dw_shape'][2]
        out_f = params['out_f']
        multiplier = params['dw_shape'][3]
        padding = params['padding']
        stride = params['stride']
        dw_init_type = params['dw_init_type']
        pw_init_type = params['pw_init_type']
        use_bias = params['use_bias']
        activation = ActivationConverter.str_to_activation(params['activation'])
        return SeparableConvLayer(
            kw=kw, kh=kh, in_f=in_f, out_f=out_f, multiplier=multiplier,
            padding=padding, stride=stride, activation=activation,
            dw_kernel_initializer=dw_init_type, pw_kernel_initializer=pw_init_type,
            use_bias=use_bias, name=name
        )

    @staticmethod
    def __atrousconv_layer_from_dict(params):
        name = params['name']
        kw = params['shape'][0]
        kh = params['shape'][1]
        in_f = params['shape'][2]
        out_f = params['shape'][3]
        rate = params['rate']
        padding = params['padding']
        init_type = params['init_type']
        use_bias = params['use_bias']
        activation = ActivationConverter.str_to_activation(params['activation'])
        return AtrousConvLayer(
            kw=kw, kh=kh, in_f=in_f, out_f=out_f, rate=rate,
            padding=padding, activation=activation,
            kernel_initializer=init_type,
            use_bias=use_bias, name=name
        )

    @staticmethod
    def __dense_layer_from_dict(params):
        name = params['name']
        input_shape = params['input_shape']
        output_shape = params['output_shape']
        activation = ActivationConverter.str_to_activation(params['activation'])
        init_type = params['init_type']
        use_bias = params['use_bias']
        return DenseLayer(
            in_d=input_shape, out_d=output_shape,
            activation=activation, name=name,
            mat_initializer=init_type, use_bias=use_bias
        )

    @staticmethod
    def __batchnorm_layer_from_dict(params):
        name = params['name']
        D = params['D']
        decay = params['decay']
        eps = params['eps']
        use_beta = params['use_beta']
        use_gamma = params['use_gamma']
        return BatchNormLayer(D=D, name=name, decay=decay, eps=eps,
                              use_beta=use_beta, use_gamma=use_gamma)

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
    def __mulbyalpha_layer_from_dict(params):
        name = params['name']
        alpha = params['alpha']
        return MulByAlphaLayer(alpha=alpha, name=name)

    @staticmethod
    def __maxpool_layer_from_dict(params):
        return MaxPoolLayer(**params)

    @staticmethod
    def __avgpool_layer_from_dict(params):
        ksize = params['ksize']
        strides = params['strides']
        padding = params['padding']
        name = params['name']
        return AvgPoolLayer(
            ksize=ksize, strides=strides,
            padding=padding, name=name
        )

    @staticmethod
    def __activation_layer_from_dict(params):
        activation = ActivationConverter.str_to_activation(params['activation'])
        name = params['name']
        return ActivationLayer(activation=activation, name=name)

    @staticmethod
    def __dropout_layer_from_dict(params):
        p_keep = params['p_keep']
        name = params['name']
        noise_shape = params['noise_shape']
        seed = params['seed']
        rate = params['rate']
        return DropoutLayer(p_keep=p_keep, name=name, noise_shape=noise_shape,
                            seed=seed, rate=rate)

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
    def __reshape_layer_from_dict(params):
        name = params['name']
        new_shape = params['new_shape']
        return ReshapeLayer(
            new_shape=new_shape,
            name=name
        )

    # -----------------------------------------------------------GRAPH RESTORATION--------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def restore_graph(outputs, graph_info_json, batch_sz=None):
        # dict {NameTensor : Info about this tensor}
        graph_info = {}

        for tensor in graph_info_json:
            graph_info[tensor['name']] = tensor

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
                    temp.update({
                        'type': parent_layer_info['type'],
                        'params': parent_layer_info['params']}
                    )
                    if batch_sz is not None:
                        temp['params']['input_shape'][0] = batch_sz
                    answer = Builder.__layer_from_dict(temp)

                coll_tensors[from_] = answer
                return answer
            else:
                return coll_tensors[from_]

        for name_output in outputs:
            restore_in_and_out_x(name_output)

        return coll_tensors
