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

from sklearn.utils import shuffle
import tensorflow as tf
from tqdm import tqdm

from scipy.special import binom

from makiflow.base import MakiTensor
from makiflow.generators.segmentator import SegmentIterator
from makiflow.layers import InputLayer
from makiflow.base.maki_entities import MakiCore


class SegmentatorBasic(MakiCore):
    DEFAULT_NAME = 'MakiSegmentator'
    INPUT_MT = 'input_mt'
    OUTPUT_MT = 'output_mt'
    NAME = 'name'
    LABELS = 'labels'

    def __init__(self, input_s: InputLayer, output: MakiTensor, name=DEFAULT_NAME):
        self.name = str(name)
        graph_tensors = output.get_previous_tensors()
        graph_tensors.update(output.get_self_pair())
        super().__init__(graph_tensors, outputs=[output], inputs=[input_s])
        self._training_vars_are_ready = False

    def predict(self, x):
        return self._session.run(
            self._output_data_tensors[0],
            feed_dict={self._input_data_tensors[0]: x}
        )

    def _get_model_info(self):
        return {
            SegmentatorBasic.NAME: self.name,
            SegmentatorBasic.INPUT_MT: self._inputs[0].get_name(),
            SegmentatorBasic.OUTPUT_MT: self._outputs[0].get_name()
        }

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------SETTING UP TRAINING-----------------------------------------

    # noinspection PyAttributeOutsideInit
    def set_generator(self, generator):
        self._generator = generator
        if not self._set_for_training:
            super()._setup_for_training()
        if not self._training_vars_are_ready:
            self._prepare_training_vars(use_generator=True)

    def _prepare_training_vars(self, use_generator=False):
        out_shape = self._outputs[0].get_shape()
        self.out_w = out_shape[1]
        self.out_h = out_shape[2]
        self.total_predictions = out_shape[1] * out_shape[2]
        self.num_classes = out_shape[-1]
        self.batch_sz = out_shape[0]

        # If generator is used, then the input data tensor will by an image tensor
        # produced by the generator, since it's the input layer.
        self._images = self._input_data_tensors[0]
        if use_generator:
            self._labels = self._generator.get_iterator()[SegmentIterator.MASK]
        else:
            self._labels = tf.placeholder(tf.int32, shape=out_shape[:-1], name=SegmentatorBasic.LABELS)

        training_out = self._training_outputs[0]
        self._flattened_logits = tf.reshape(training_out, shape=[-1, self.total_predictions, self.num_classes])
        self._flattened_labels = tf.reshape(self._labels, shape=[-1, self.total_predictions])

        self._ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self._flattened_logits, labels=self._flattened_labels
        )

        self._training_vars_are_ready = True
        self._use_generator = use_generator


