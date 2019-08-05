from makiflow.layers import InputLayer, ConvLayer, MaxPoolLayer, FlattenLayer, DenseLayer
import requests
import shutil
import os
import tensorflow as tf

BACKBONE = {
    'vgg16': ['https://drive.google.com/uc?export=download&id=1AF91M_MMSO-Enfi8E9AJPZD4KPbru2DG',
              'https://drive.google.com/uc?export=download&id=17D75ZhiNAhOI7AJqmUkb_TSVsf_V2ruJ',
              'https://drive.google.com/uc?export=download&id=1zr1avIJ3LMEiN_YANycniNA8XhVD1t4S',
              'https://drive.google.com/uc?export=download&id=1--Rls_uo5hH0J4t6duPMwWlSiOCprDB8'],

    'vgg19': 'https://drive.google.com/open?id=1ug0v1ApPLyk2LSsmTXAMNTAQP1642w_K',
}


class PretrainedWeightsProvider:

    def get_pretrained_backbone(self, backbone_name: str, _input_shape: list or tuple):
        return getattr(self, f'__get_pretrained_backbone_{backbone_name}', _input_shape)

    def __get_pretrained_backbone_vgg19(self, _input_shape):
        tf.logging.info(f'VGG19 is unsupported, but you can download weights manually:{BACKBONE["vgg16"]}')

    def __get_pretrained_backbone_vgg16(self, _input_shape):
        in_x = InputLayer(input_shape=_input_shape, name='Input')
        x = ConvLayer(kw=3, kh=3, in_f=3, out_f=64, name='block1_conv1')(in_x)
        x = ConvLayer(kw=3, kh=3, in_f=64, out_f=64, name='block1_conv2')(x)
        x = MaxPoolLayer(name='block1_pool')(x)

        x = ConvLayer(kw=3, kh=3, in_f=64, out_f=128, name='block2_conv1')(x)
        x = ConvLayer(kw=3, kh=3, in_f=128, out_f=128, name='block2_conv2')(x)
        x = MaxPoolLayer(name='block2_pool')(x)

        x = ConvLayer(kw=3, kh=3, in_f=128, out_f=256, name='block3_conv1')(x)
        x = ConvLayer(kw=3, kh=3, in_f=256, out_f=256, name='block3_conv2')(x)
        x = ConvLayer(kw=3, kh=3, in_f=256, out_f=256, name='block3_conv3')(x)
        x = MaxPoolLayer(name='block3_pool')(x)

        x = ConvLayer(kw=3, kh=3, in_f=256, out_f=512, name='block4_conv1')(x)
        x = ConvLayer(kw=3, kh=3, in_f=512, out_f=512, name='block4_conv2')(x)
        x = ConvLayer(kw=3, kh=3, in_f=512, out_f=512, name='block4_conv3')(x)
        x = MaxPoolLayer(name='block4_pool')(x)

        x = ConvLayer(kw=3, kh=3, in_f=512, out_f=512, name='block5_conv1')(x)
        x = ConvLayer(kw=3, kh=3, in_f=512, out_f=512, name='block5_conv2')(x)
        x = ConvLayer(kw=3, kh=3, in_f=512, out_f=512, name='block5_conv3')(x)
        x = MaxPoolLayer(name='block5_pool')(x)

        x = FlattenLayer(name='flatten')(x)
        x = DenseLayer(in_d=25088, out_d=4096, name='fc1')(x)
        x = DenseLayer(in_d=4096, out_d=4096, name='fc2')(x)

        net_layers_names = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1',
                            'block3_conv2',
                            'block3_conv3', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1',
                            'block5_conv2',
                            'block5_conv3', 'fc1', 'fc2']

        net_name = 'vgg16'
        to_weights_path = self.__get_to_weights_path(net_name)
        if to_weights_path is not None:
            return in_x, x, net_layers_names, to_weights_path
        else:
            return None

    def __get_to_weights_path(self, net_name):
        to_weights_path = f'weights/vgg16/model_{net_name}.ckpt'

        if not os.path.exists(to_weights_path + '.meta'):
            file_names = [
                'checkpoint',
                f'model_{net_name}.ckpt.data-00000-of-00001',
                f'model_{net_name}.ckpt.index',
                f'model_{net_name}.ckpt.meta'
            ]
            is_success = self.__download_weights(net_name, file_names)
            if is_success:
                return to_weights_path
            else:
                None
        return to_weights_path

    def __download_weights(self, name, file_names) -> bool:
        try:
            path_to_download_prefix = f'weights/{name}/'
            for index, url in enumerate(BACKBONE[name]):
                path = path_to_download_prefix + file_names[index]
                with requests.get(url, stream=True) as req:
                    with open(path, 'wb') as f:
                        shutil.copyfileobj(req.raw, f)
            return True
        except Exception as ex:
            return False
