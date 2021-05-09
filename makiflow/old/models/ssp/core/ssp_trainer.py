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

from abc import ABC, abstractmethod
from makiflow.core import MakiTrainer
from .head_label import HeadLabel
from .head import Head
import tensorflow as tf
import numpy as np


class SSPTrainer(MakiTrainer, ABC):
    # Entity types
    COORDINATES = 'Coordinates'
    POINT_VISIBILITY_INDICATORS = 'PointVisibilityIndicators'
    HUMAN_PRESENCE_INDICATORS = 'HumanPresenceIndicators'
    HEAD = 'HEAD'

    COORDS_LOSS = 'CoordsLoss'
    POINT_INDICATORS_LOSS = 'PointIndicatorsLoss'
    HUMAN_INDICATORS_LOSS = 'HumanIndicatorsLoss'

    @staticmethod
    def encode(entity_type, feature_map_size, bbox_config):
        h, w = feature_map_size
        h_scale, w_scale = bbox_config
        return f'{entity_type}_WH{h}-{w}_BC{w_scale}-{h_scale}'

    @staticmethod
    def coordinates_name(feature_map_size, bbox_config):
        return SSPTrainer.encode(SSPTrainer.COORDINATES, feature_map_size, bbox_config)

    @staticmethod
    def point_visibility_indicators_name(feature_map_size, bbox_config):
        return SSPTrainer.encode(SSPTrainer.POINT_VISIBILITY_INDICATORS, feature_map_size, bbox_config)

    @staticmethod
    def human_presence_indicators_name(feature_map_size, bbox_config):
        return SSPTrainer.encode(SSPTrainer.HUMAN_PRESENCE_INDICATORS, feature_map_size, bbox_config)

    @staticmethod
    def decode(tensor_name):
        tensor_info = tensor_name.split('_')
        assert len(tensor_info) == 3, 'Unknown tensor_name structure. The following structure must ' \
                                      'be used: tensortype_WH=h-w_BC=w_scale-h_scale. ' \
                                      f'Received {tensor_name}'

        entity_type, feature_map_size, bbox_config = tensor_info
        allowed_entity_types = [
            SSPTrainer.COORDINATES,
            SSPTrainer.POINT_VISIBILITY_INDICATORS,
            SSPTrainer.HUMAN_PRESENCE_INDICATORS
        ]
        assert entity_type in allowed_entity_types, 'Unknown entity type.' \
                                                    f'Allowed entity types: {allowed_entity_types}. ' \
                                                    f'Received: {entity_type}'

        feature_map_size = feature_map_size.replace('WH', '')
        h, w = feature_map_size.split('-')
        h, w = int(h), int(w)

        bbox_config = bbox_config.replace('BC', '')
        w_scale, h_scale = bbox_config.split('-')
        w_scale, h_scale = float(w_scale), float(h_scale)
        return entity_type, (h, w), (w_scale, h_scale)

    def _setup_label_placeholders(self):
        print('Setting up label placeholders.')
        print('Be aware that in this case the order of the training tensors corresponds to the'
              'order of heads in the model. Example:'
              'head0.coords, head0.point_indicators, head0.human_indicators, head1.coords, ...'
              )

        heads = super().get_model().get_heads()
        batch_size = super().get_batch_size()

        label_placeholders = {}
        for head in heads:
            point_indicators_shape = head.get_point_indicators().shape()
            n_points = point_indicators_shape[-1]
            grid_size = head.get_grid_size()
            bbox_config = head.get_bbox_configuration()

            coords_shape = [batch_size, *grid_size, n_points * 2]
            point_indicators_shape = [batch_size, *grid_size, n_points]
            human_indicators_shape = [batch_size, *grid_size, 1]

            coords_name = SSPTrainer.coordinates_name(grid_size, bbox_config)
            point_indicators_name = SSPTrainer.point_visibility_indicators_name(grid_size, bbox_config)
            human_indicators_name = SSPTrainer.human_presence_indicators_name(grid_size, bbox_config)

            label_placeholders[coords_name] = tf.placeholder(
                'float32', shape=coords_shape, name=coords_name
            )
            label_placeholders[point_indicators_name] = tf.placeholder(
                'float32', shape=point_indicators_shape, name=point_indicators_name
            )
            label_placeholders[human_indicators_name] = tf.placeholder(
                'float32', shape=human_indicators_shape, name=human_indicators_name
            )

        return label_placeholders

    def _init(self):
        super()._init()
        self._collect_model_heads()
        self._setup_head_labels()
        model_heads = super().get_model().get_heads()
        assert len(model_heads) == len(self._head_labels), "The number of model's heads and the head labels must " \
                                                           f"be the the sum but N_model_heads={len(model_heads)} and " \
                                                           f"N_head_labels={len(self._head_labels)}"

    def _collect_model_heads(self):
        """
        Collects model's heads into a dictionary:
        { grid_size: [head1, head2, ...] }.
        This dictionary is then used to retrieve model's head with a configuration
        similar to a particular HeadLabel.
        """
        model_heads = super().get_model().get_heads()

        self._model_heads_collections = {}
        for head in model_heads:
            grid_size = head.get_grid_size()
            heads_collection = self._model_heads_collections.get(tuple(grid_size), [])
            heads_collection.append(head)

            self._model_heads_collections[tuple(grid_size)] = heads_collection

    def _setup_head_labels(self):
        label_tensors = super().get_label_tensors()

        tensor_collections = {
            SSPTrainer.COORDINATES: {},
            SSPTrainer.POINT_VISIBILITY_INDICATORS: {},
            SSPTrainer.HUMAN_PRESENCE_INDICATORS: {}
        }

        for label_name, tensor in label_tensors.items():
            tensor_type, feature_map_size, bbox_config = SSPTrainer.decode(label_name)
            collection = tensor_collections[tensor_type]
            configuration = (*feature_map_size, *bbox_config)
            collection[configuration] = tensor

        print(tensor_collections)
        self._head_labels = []
        for configuration, coordinates in tensor_collections[SSPTrainer.COORDINATES].items():
            point_indicators = tensor_collections[SSPTrainer.POINT_VISIBILITY_INDICATORS].get(configuration)
            assert point_indicators is not None, f'coordinates tensor with configuration={configuration} does not ' \
                f'have corresponding point_indicators with the same configuration.'

            human_indicators = tensor_collections[SSPTrainer.HUMAN_PRESENCE_INDICATORS].get(configuration)
            assert human_indicators is not None, f'coordinates tensor with configuration={configuration} does not ' \
                f'have corresponding human_indicators with the same configuration.'

            head = HeadLabel(coordinates, point_indicators, human_indicators, configuration)
            self._head_labels.append(head)

    def _build_loss(self):
        losses = []
        for head_label in self._head_labels:
            head = self.find_similar_model_head(head_label)

            print('\nLinking the following heads: ')
            print(f'- label - {head_label.get_description()}')
            print(f'- nn head - {head.get_description()}')
            coords = head.get_coords()
            coords = super().get_traingraph_tensor(coords.name)
            _, ch, cw, d = coords.shape().as_list()
            # Transform the coords to the image plane.
            input_image = super().get_train_inputs_list()[0]
            _, h, w, _ = input_image.shape()
            scale = np.array([w / 2, h / 2], dtype='float32')
            flatten = lambda t: tf.reshape(t, shape=[-1, ch, cw, d])
            unflatten = lambda t: tf.reshape(t, shape=[-1, ch, cw, d // 2, 2])

            coords = unflatten(coords)
            coords = coords * scale + scale
            coords = flatten(coords)

            point_indicators = head.get_point_indicators()
            point_indicators = super().get_traingraph_tensor(point_indicators.name)
            human_indicators = head.get_human_indicators()
            human_indicators = super().get_traingraph_tensor(human_indicators.name)

            label_coords = head_label.get_coords()
            label_point_indicators = head_label.get_point_indicators()
            label_human_indicators = head_label.get_human_indicators()

            coords_loss, point_indicators_loss, human_indicators_loss = self._build_head_losses(
                coords=coords,
                point_indicators=point_indicators,
                human_indicators=human_indicators,
                label_coords=label_coords,
                label_point_indicators=label_point_indicators,
                label_human_indicators=label_human_indicators
            )
            # ----------------------------------Track coords loss----------------------------------------------
            loss_name = SSPTrainer.encode(
                SSPTrainer.COORDS_LOSS,
                head.get_grid_size(),
                head.get_bbox_configuration()
            )
            super().track_loss(coords_loss, loss_name)
            # ----------------------------Track point indicators loss------------------------------------------
            loss_name = SSPTrainer.encode(
                SSPTrainer.POINT_INDICATORS_LOSS,
                head.get_grid_size(),
                head.get_bbox_configuration()
            )
            super().track_loss(point_indicators_loss, loss_name)
            # -----------------------------Track human indicators loss-----------------------------------------
            loss_name = SSPTrainer.encode(
                SSPTrainer.HUMAN_INDICATORS_LOSS,
                head.get_grid_size(),
                head.get_bbox_configuration()
            )
            super().track_loss(human_indicators_loss, loss_name)
            print([coords_loss, point_indicators_loss, human_indicators_loss])
            losses += [coords_loss, point_indicators_loss, human_indicators_loss]

        return tf.add_n(losses)

    def find_similar_model_head(self, head_label: HeadLabel) -> Head:
        """
        Finds similar model's head with the same grid size and closest bbox configuration.

        Parameters
        ----------
        head_label : HeadLabel
            HeadLabel for which to find corresponding model's head.

        Returns
        -------
        Head
            The most similar model's head.
        """
        grid_size = tuple(head_label.get_grid_size())
        model_heads = self._model_heads_collections.get(grid_size)
        assert model_heads is not None and len(model_heads) != 0, f"Tried to find heads with grid_size={grid_size}, " \
            f"but there are no heads with such grid_size."

        l_w_scale, l_h_scale = head_label.get_bbox_configuration()
        last_similarity = 1e4
        last_head = None
        for head in model_heads:
            h_w_scale, h_h_scale = head.get_bbox_configuration()
            similarity = abs(h_w_scale - l_w_scale) + abs(h_h_scale - l_h_scale)
            if similarity < last_similarity:
                last_head = head
                last_similarity = similarity

        assert last_head is not None, 'Somehow did not found similar head.'
        return last_head

    @abstractmethod
    def _build_head_losses(
            self,
            coords, point_indicators, human_indicators,
            label_coords, label_point_indicators, label_human_indicators
    ) -> tuple:
        pass

    def get_label_feed_dict_config(self):
        label_tensors = super().get_label_tensors()

        label_feed_dict = {}
        for i, (name, tensor) in enumerate(label_tensors.items()):
            label_feed_dict[tensor] = i

        return label_feed_dict








