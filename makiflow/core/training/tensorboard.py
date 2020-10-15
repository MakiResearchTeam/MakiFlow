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
from .athena import Athena
from abc import ABC
import tensorflow as tf


class TensorBoard:
    def __init__(self):
        self._tb_is_setup = False
        self._tb_writer = None
        # Counter for total number of training iterations.
        self._tb_summaries = []
        self._total_summary = None

    def set_tensorboard_writer(self, writer):
        """
        Creates logging file for the Tensorboard in the given `logdir_path` directory.
        Parameters
        ----------
        writer : tf.FileWriter
            Path to the log directory.
        """
        self._tb_writer = writer

    def add_scalar(self, scalar, name):
        summary = tf.summary.scalar(name, scalar)
        self.add_summary(summary)

    def add_histogram(self, tensor, name):
        summary = tf.summary.histogram(name, tensor)
        self.add_summary(summary)

    def add_summary(self, summary):
        self._tb_summaries.append(summary)

    def set_layers_histograms(self, layer_names):
        # noinspection PyAttributeOutsideInit
        self._layers_histograms = layer_names

    def close_tensorboard(self):
        """
        Closes the logging writer for the Tensorboard
        """
        self._tb_writer.close()

    def setup_tensorboard(self):
        assert len(self._tb_summaries) != 0, 'No summaries found.'
        print('Collecting histogram tensors...')

        # Collect all layers histograms
        for layer_name in self._layers_histograms:
            # Add weights histograms
            with tf.name_scope(f'{layer_name}/weight'):
                for weight in self._layer_weights[layer_name]:
                    self.add_summary(tf.summary.histogram(name=weight.name, values=weight))

            # Add grads histograms
            with tf.name_scope(f'{layer_name}/grad'):
                for weight in self._layer_weights[layer_name]:
                    grad = self._var2grad.get(weight)
                    if grad is None:
                        print(f'Did not find gradient for layer={layer_name}, var={weight.name}')
                        continue
                    self.add_summary(tf.summary.histogram(name=weight.name, values=grad))

        self._total_summary = tf.summary.merge(self._tb_summaries)
        self._tb_is_setup = True

    def get_total_summary(self):
        assert self._total_summary is not None, 'The tensorboard is not setup.'
        return self._total_summary

    def is_setup(self):
        return self._tb_is_setup
