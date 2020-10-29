# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

import tensorflow as tf
from makiflow.core.graph_entities import MakiTensor
from makiflow.core.inference import MakiModel as MakiCore

from makiflow.generators.nn_render import NNRIterator


class NeuralRenderBasis(MakiCore):

    INPUT_MT = 'input_mt'
    OUTPUT_MT = 'output_mt'
    NAME = 'name'
    IMAGES = 'images'

    @staticmethod
    def from_json(path_to_model):
        # TODO
        pass

    def __init__(self, input_x, output_x, sampled_texture: MakiTensor, name='NeuralRenderModel'):
        """
        Create NeuralRender model which provides API to train and tests different models for neural rendering

        Parameters
        ----------
        input_x : MakiTensor
            Input MakiTensor
        output_x : MakiTensor
            Output MakiTensor
        sampled_texture : MakiTensor
            Sampled texture of the model, which is used for building a loss to force produce RBG texture
        name : str
            Name of this model
        """
        self.name = str(name)
        graph_tensors = output_x.get_previous_tensors()
        graph_tensors.update(output_x.get_self_pair())
        super().__init__(outputs=[output_x], inputs=[input_x])
        self._sampled_texture = sampled_texture

        self._training_vars_are_ready = False
        self._learn_rgb_texture = False
        self._sep_loss = None
        self._generator = None

    def predict(self, x):
        """
        Get result from neural network according to certain input

        Parameters
        ----------
        x: ndarray
            Input for neural network, i. e. for this model.

        Returns
        ----------
        ndarray
            Output of the neural network
        """
        return self._session.run(
            self._output_data_tensors[0],
            feed_dict={self._input_data_tensors[0]: x}
        )

    def _get_model_info(self):
        return {
            NeuralRenderBasis.NAME: self.name,
            NeuralRenderBasis.INPUT_MT: self._inputs[0].get_name(),
            NeuralRenderBasis.OUTPUT_MT: self._outputs[0].get_name()
        }

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------SETTING UP TRAINING-------------------------------------

    def _prepare_training_vars(self):
        if not self._set_for_training:
            super()._setup_for_training()
        # VARIABLES PREPARATION
        out_shape = self._outputs[0].get_shape()
        self._out_h = out_shape[1]
        self._out_w = out_shape[2]
        self._batch_sz = out_shape[0]

        self._uv_maps = self._input_data_tensors[0]
        if self._generator is not None:
            self._images = self._generator.get_iterator()[NNRIterator.IMAGE]
        else:
            self._images = tf.placeholder(tf.float32, shape=out_shape, name=NeuralRenderBasis.IMAGES)
        self._training_out = self._training_outputs[0]

        # OTHER PREPARATIONS
        if self._learn_rgb_texture:
            self._build_texture_loss()
        self._training_vars_are_ready = True

    def set_generator(self, generator):
        """
        Set generator (i. e. pipeline) for this model

        Parameters
        ----------
        generator : mf.generators
            Certain generator for this model
        """
        self._generator = generator

    # noinspection PyAttributeOutsideInit
    def set_learn_rgb_texture(self, scale):
        """
        Force the neural texture to learn RGB values in the first 3 channels.

        Parameters
        ----------
        scale : float
            Final loss will be derived in the following way:
            final_loss = objective + scale * texture_loss.
        """
        self._learn_rgb_texture = True
        self._texture_loss_scale = scale

    def _build_texture_loss(self):
        texture_tensor = self._sampled_texture.get_data_tensor()
        # [batch_size, height, width, channels]
        sampled_rgb_channels = texture_tensor[:, :, :, :3]
        diff = tf.abs(sampled_rgb_channels - self._images)
        self._texture_loss = tf.reduce_mean(diff)

    def _build_final_loss(self, training_loss):
        # Override the method for the later ease of loss building
        if self._learn_rgb_texture:
            training_loss = training_loss + self._texture_loss * self._texture_loss_scale
        if self._sep_loss is not None:
            training_loss = training_loss + self._sep_loss
        loss = super()._build_final_loss(training_loss)
        return loss

    def add_sep_loss(self, loss):
        """
        Used for adding other terms to the final loss. The VGG loss can be such a term.
        Parameters
        ----------
        loss : tf.Tensor
            A scalar of the additional loss.
        """
        self._sep_loss = loss

    def get_output_images(self):
        """
        This method is meant to be used for creation of separate losses.
        WARNING! This method must be called after `set_learn_rgb_texture` and
        `set_generator`.
        Returns
        -------
        tf.Tensor
            Tensor of the output images generated by the renderer.
        """
        if not self._training_vars_are_ready:
            self._prepare_training_vars()
        return self._training_out


