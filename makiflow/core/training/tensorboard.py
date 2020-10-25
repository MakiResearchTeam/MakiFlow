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


class TensorBoard:
    def __init__(self):
        self._tb_is_setup = False
        self._tb_writer = None
        # Counter for total number of training iterations.
        self._tb_summaries = []
        self._total_summary = None
        # Counter of iterations
        self._counter = 0

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

    def close_tensorboard(self):
        """
        Closes the logging writer for the Tensorboard
        """
        self._tb_writer.close()

    def setup_tensorboard(self):
        assert len(self._tb_summaries) != 0, 'No summaries found.'
        print('Collecting histogram tensors...')

        self._total_summary = tf.summary.merge(self._tb_summaries)
        self._tb_is_setup = True

    def get_total_summary(self):
        assert self._total_summary is not None, 'The tensorboard is not setup.'
        return self._total_summary

    def is_setup(self):
        return self._tb_is_setup

    def increment(self):
        # Must be called during each iteration of the training cycle
        self._counter += 1

    def write_summary(self, summary):
        """
        Writes the summary to the Tensorboard. If the tensorboard writer is not provided, does nothing.
        Parameters
        ----------
        summary : tf.Summary
            The total summary (received from the `get_total_summary` method).
        Returns
        -------

        """
        if self._tb_writer is None:
            return
        self._tb_writer.add_summary(summary, self._counter)
