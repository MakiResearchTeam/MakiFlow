from makiflow.layers.sf_layer import SimpleForwardLayer
from makiflow.models.nn_render.utils import grid_sample
import tensorflow as tf
import numpy as np


class TParams:
    WIDTH = 'WIDTH'
    HEIGHT = 'HEIGHT'
    NUM_F = 'NUM_F'
    DEPTH = 'DEPTH'
    NAME = 'NAME'


class SingleTextureLayer(SimpleForwardLayer):
    def __init__(self, width, height, num_f, name, text_init=None):
        self._w = width
        self._h = height
        self._num_f = num_f

        if text_init is None:
            text_init = np.random.randn(1, height, width, num_f).astype(np.float32)
        self._text_name = f'NTexture{width}_{height}_{name}'
        self._texture = tf.Variable(text_init, name=self._text_name)
        params = [self._texture]
        named_params_dict = {self._text_name: self._texture}
        super().__init__(name, params, named_params_dict)

    def _forward(self, x):
        # Normalize the input UV map so that its coordinates are within [-1, 1] range.
        x = x * 2.0 - 1.0
        batch_size = x.get_shape().as_list()[0]
        expanded_texture = tf.concat([self._texture] * batch_size, axis=0)
        return grid_sample(expanded_texture, x)

    def _training_forward(self, x):
        return self._forward(x)

    def to_dict(self):
        return {
            'type': 'SingleTextureLayer',
            'params': {
                TParams.WIDTH: self._w,
                TParams.HEIGHT: self._h,
                TParams.NUM_F: self._num_f,
                TParams.NAME: self._name
            }
        }


class LaplacianPyramidTextureLayer(SimpleForwardLayer):
    def __init__(self, width, height, num_f, depth, name, text_init=None):
        self._w = width
        self._h = height
        self._num_f = num_f
        self._depth = depth

        if text_init is None:
            text_init = [None] * depth

        self._textures = []
        params = []
        named_params_dict = {}
        for d in range(depth):
            texture = SingleTextureLayer(
                width=width // 2**d,
                height=height // 2**d,
                num_f=num_f,
                name=name+str(d),
                text_init=text_init[d]
            )
            self._textures += [texture]
            params += texture.get_params()
            named_params_dict.update(texture.get_params_dict())

        super().__init__(name, params, named_params_dict)

    # noinspection PyProtectedMember
    def _forward(self, x):
        # Normalize the input UV map so that its coordinates are within [-1, 1] range.
        y = []
        for d in range(self._depth):
            y += [self._textures[d]._forward(x)]
        return tf.add_n(y)

    def _training_forward(self, x):
        return self._forward(x)

    def to_dict(self):
        return {
            'type': 'SingleTextureLayer',
            'params': {
                TParams.WIDTH: self._w,
                TParams.HEIGHT: self._h,
                TParams.NUM_F: self._num_f,
                TParams.DEPTH: self._depth,
                TParams.NAME: self._name,
            }
        }
