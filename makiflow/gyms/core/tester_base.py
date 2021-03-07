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
from abc import ABC, abstractmethod
import os


class TesterBase(ABC):
    # Using in conjugation with trainer.
    # After the model was trained for some time, call the evaluation
    # method and all the info will recorded to the tensorboard.
    TEST_CONFIG = 'test_config'
    TB_FOLDER = 'tb_folder'  # folder for tensorboard to write data in
    TEST_IMAGE = 'test_image'
    BATCH_SIZE = 'batch_size'
    NORMALIZATION_SHIFT = 'normalization_shift'
    NORMALIZATION_DIV = 'normalization_div'
    NORM_MODE = 'norm_mode'
    USE_BGR2RGB = 'use_bgr2rgb'
    RESIZE_TO = 'resize_to'

    def __init__(self, config: dict, sess):
        self._config = config[TesterBase.TEST_CONFIG]
        self._tb_writer = tf.summary.FileWriter(config[TesterBase.TB_FOLDER])
        self._sess = sess

        # Image preprocess stuff
        self._norm_div = self._config[self.NORMALIZATION_DIV]
        self._norm_shift = self._config[self.NORMALIZATION_SHIFT]
        self._norm_mode = self._config[self.NORM_MODE]
        self._resize_to = self._config[self.RESIZE_TO]

        self._use_bgr2rgb = self._config[self.USE_BGR2RGB]
        # The summaries to write
        self._summaries = {}
        # Placeholder that take in the data for the summary
        self._summary_inputs = {}

        self._init()

    def _init(self):
        pass

    def add_image(self, name, n_images=1):
        """
        Adds an image summary to the tensorboard.
        The image dtype must by uint8 and have shape (batch_size, h, w, c).

        Parameters
        ----------
        name : str
            Name that will be displayed on the tensorboard.
        n_images : int
            Maximum number of images to display on the board.
        """
        image = tf.placeholder(dtype=tf.uint8)
        self._summary_inputs.update(
            {name: image}
        )
        image_summary = tf.summary.image(name, image, max_outputs=n_images)
        self._summaries.update(
            {name: image_summary}
        )

    def add_scalar(self, name):
        """
        Adds a scalar summary (e.g. accuracy) to the tensorboard.
        The image dtype must by float32.

        Parameters
        ----------
        name : str
            Name that will be displayed on the tensorboard.
        """
        scalar = tf.placeholder(dtype=tf.float32)
        self._summary_inputs.update(
            {name: scalar}
        )
        scalar_summary = tf.summary.scalar(name, scalar)
        self._summaries.update(
            {name: scalar_summary}
        )

    def write_summaries(self, summaries, step=None):
        """
        Writes the summary to the tensorboard log file.
        Parameters
        ----------
        summaries : dict
            Contains pairs (name, data). `data` can be whether scalar or image.
        step : int
            The training/evaluation step number.
        """
        for summary_name in summaries:
            data = summaries[summary_name]
            s_input = self._summary_inputs[summary_name]
            summary = self._summaries[summary_name]

            summary_tensor = self._sess.run(
                summary,
                feed_dict={
                    s_input: data
                }
            )
            self._tb_writer.add_summary(summary_tensor, global_step=step)
        # self._tb_writer.flush()

    @abstractmethod
    def evaluate(self, model, iteration, path_save_res):
        pass

    def get_writer(self):
        return self._tb_writer

    @abstractmethod
    def final_eval(self, path_to_save):
        pass
