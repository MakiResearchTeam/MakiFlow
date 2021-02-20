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


class TensorBoardV2:
    def __init__(self):
        self._tb_writer = None
        self._session = None
        # { name : tf.Summary }
        self._python_summaries = {}
        # Required to create a summary object via session call
        # { name: tf.placeholder }
        self._python_summary_placeholders = {}
        self._summaries = {}

    def add_summary(self, summary: tf.Summary, name: str):
        assert not (name in self._summaries), f'Summary with name={name} already exists.'
        self._summaries[name] = summary

    def set_writer(self, writer):
        if isinstance(writer, str):
            self._tb_writer = tf.summary.FileWriter(writer)
        elif isinstance(writer, tf.summary.FileWriter):
            self._tb_writer = writer

    def set_session(self, session: tf.Session):
        self._session = session

    # --- TensorFlow summaries

    def add_scalar(self, scalar, name):
        summary = tf.summary.scalar(name, scalar)
        self.add_summary(summary, name)

    def add_histogram(self, tensor, name):
        summary = tf.summary.histogram(name, tensor)
        self.add_summary(summary, name)

    def add_image(self, tensor, name):
        summary = tf.summary.image(name, tensor)
        self.add_summary(summary, name)

    # --- Python summaries

    def add_scalar_python(self, name):
        """
        Adds a scalar summary (e.g. accuracy) to the tensorboard.
        The image dtype must by float32.

        Parameters
        ----------
        name : str
            Name that will be displayed on the tensorboard.
        """
        scalar = tf.placeholder(dtype=tf.float32)
        self._python_summary_placeholders.update(
            {name: scalar}
        )
        scalar_summary = tf.summary.scalar(name, scalar)
        self.add_summary(scalar_summary, name)

    def add_histogram_python(self, name, dtype='float32'):
        tensor = tf.placeholder(dtype=dtype)
        self._python_summary_placeholders.update(
            {name: tensor}
        )
        hist_summary = tf.summary.histogram(name, tensor)
        self.add_summary(hist_summary, name)

    def add_image_python(self, name, n_images=1):
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
        self._python_summary_placeholders.update(
            {name: image}
        )
        image_summary = tf.summary.image(name, image, max_outputs=n_images)
        self.add_summary(image_summary, name)

    def update_board(self, python_data: dict = None, step=None):
        """
        Writes the summary to the tensorboard log file.

        Parameters
        ----------
        python_data : dict
            Contains pairs (name, data). `data` can be whether a scalar or an image or
            a tensor to make a histogram for.
        step : int
            The training/evaluation step number.
        """
        feed_dict = None
        # Create a feed_dict to generate summary objects from python data
        if python_data is not None:
            feed_dict = {}
            for tensor_name, data in python_data.items():
                feed_dict[self._python_summary_placeholders[tensor_name]] = data

        summary_dict = self._session.run(
            self._summaries,
            feed_dict=feed_dict
        )
        for summary_name, summary in summary_dict:
            self._tb_writer.add_summary(summary, global_step=step)
