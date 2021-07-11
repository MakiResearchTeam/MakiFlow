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

from makiflow.core.debug import d_msg
from makiflow.models.ssp.core.head_interface import HeadInterface


class HeadLabel(HeadInterface):
    def __init__(self, coords: tf.Tensor, point_indicators: tf.Tensor, human_indicators: tf.Tensor, configuration):
        """
        An entity that encapsulates all the tensors necessary to make predictions on a particular grid.
        It makes makes sure the shapes are synchronized and also collects necessary info for the trainer.

        Parameters
        ----------
        coords : tf.Tensor
            Tensor of the regressed coordinates of the skeleton points. Must lie approximately within
            the [-1, 1] interval.
        point_indicators : tf.Tensor
            Tensor of binary indicators of whether a particular point of the skeleton is visible.
        human_indicators : tf.Tensor
            Tensor of binary indicators of whether a human is present in a particular location
            of the grid.
        """
        self._context = f'SSP HeadLabel({coords.name}, {point_indicators.name}, {human_indicators.name}, {configuration})'

        self._coords = coords
        self._point_indicators = point_indicators
        self._human_indicators = human_indicators
        assert len(configuration) == 4, d_msg(
            self._context,
            f'Configuration must has length=4, received length={configuration}'
        )
        h, w, w_scale, h_scale = configuration
        self._grid_size = [h, w]
        self._bbox_config = [w_scale, h_scale]
        self.__check_dimensionality()

    def __check_dimensionality(self):
        # All have dimensions [b, h, w, c]
        coords_shape = self._coords.get_shape().as_list()
        point_indicators_shape = self._point_indicators.get_shape().as_list()
        human_indicators_shape = self._human_indicators.get_shape().as_list()

        # Only convolutional networks are supported
        assert len(coords_shape) == 4 and \
            len(point_indicators_shape) == 4 and \
            len(human_indicators_shape) == 4, d_msg(
            self._context,
            'Dimensionality of all tensors must be 4, received '
            f'dim(coords)={len(coords_shape)}, '
            f'dim(point_indicators)={len(point_indicators_shape)}, '
            f'dim(human_indicators)={len(human_indicators_shape)}'
        )

        # Check spatial shape (h, w)
        assert coords_shape[1:-1] == point_indicators_shape[1:-1] and \
            coords_shape[1:-1] == human_indicators_shape[1:-1] and \
            point_indicators_shape[1:-1] == human_indicators_shape[1:-1], d_msg(
            self._context,
            'Spatial shapes are not aligned. Received '
            f'coords_shape={coords_shape}, '
            f'point_indicators_shape={point_indicators_shape}, '
            f'human_indicators_shape={human_indicators_shape}'
        )

        # Check whether the spatial shape (h, w) is aligned with the grid_size from the configuration
        assert coords_shape[1:-1] == self._grid_size and \
            human_indicators_shape[1:-1] == self._grid_size and \
            point_indicators_shape[1:-1] == self._grid_size, d_msg(
            self._context,
            'Spatial shapes are not aligned with the grid size from the configuration. Received '
            f'coords_shape={coords_shape}, '
            f'point_indicators_shape={point_indicators_shape}, '
            f'human_indicators_shape={human_indicators_shape}, '
            f'grid_size={self._grid_size}'
        )

        # Check alignment of the number of points between coords and point indicators
        n_coords = coords_shape[-1]
        assert n_coords % 2 == 0, d_msg(
            self._context,
            f'coords must have an even number of channel, received {n_coords}.'
        )

        n_points = n_coords // 2
        assert n_points == point_indicators_shape[-1], d_msg(
            self._context,
            f'Number of points in coords and point_indicators must be the same, '
            f'received {n_points} and {point_indicators_shape[-1]}.'
        )

        # Check whether human_indicators has a single channel
        assert human_indicators_shape[-1] == 1, d_msg(
            self._context,
            f'human_indicators tensor must have 1 channel, received {human_indicators_shape[-1]}.'
        )

    def get_grid_size(self) -> list:
        return self._grid_size

    def get_bbox_configuration(self) -> list:
        return self._bbox_config

    def get_coords(self):
        return self._coords

    def get_point_indicators(self):
        return self._point_indicators

    def get_human_indicators(self):
        return self._human_indicators

    def get_description(self):
        description = self._context
        description = description + f'/GridSize={self.get_grid_size()}'
        description = description + f'/BboxConfig={self.get_bbox_configuration()}'
        return description


if __name__ == '__main__':
    # Sanity check
    coords = tf.placeholder('float32', shape=[1, 128, 128, 20])
    point_indicators = tf.placeholder('float32', shape=[1, 128, 128, 10])
    human_indicators = tf.placeholder('float32', shape=[1, 128, 128, 1])
    HeadLabel(coords, point_indicators, human_indicators, (128, 128, 1, .5))

    from makiflow.core.debug import DebugContext

    with DebugContext('Checking spatial shape.'):
        coords = tf.placeholder('float32', shape=[1, 129, 128, 20])
        HeadLabel(coords, point_indicators, human_indicators, (128, 128, 1, .5))

    with DebugContext('Checking number of points.'):
        coords = tf.placeholder('float32', shape=[1, 128, 128, 18])
        HeadLabel(coords, point_indicators, human_indicators, (128, 128, 1, .5))

    with DebugContext('Checking number of human indicators.'):
        coords = tf.placeholder('float32', shape=[1, 128, 128, 20])
        human_indicators = tf.placeholder('float32', shape=[1, 128, 128, 2])
        HeadLabel(coords, point_indicators, human_indicators, (128, 128, 1, .5))

    with DebugContext('Checking alignment with the grid size from the configuration.'):
        human_indicators = tf.placeholder('float32', shape=[1, 128, 128, 1])
        HeadLabel(coords, point_indicators, human_indicators, (128, 127, 1, .5))
