from makiflow.core import MakiTrainer
import tensorflow as tf
from abc import ABC


class ClassificatorTrainer(MakiTrainer, ABC):
    WEIGHT_MAP = 'WEIGHT_MAP'
    LABELS = 'LABELS'

    def _init(self):
        super()._init()
        logits_makitensor = super().get_model().get_logits()
        self._logits_name = logits_makitensor.get_name()
        self._num_classes = logits_makitensor.get_shape()[-1]
        self._labels = super().get_label_tensors()[ClassificatorTrainer.LABELS]
        self._weight_map = super().get_label_tensors()[ClassificatorTrainer.WEIGHT_MAP]

    def get_labels(self):
        return self._labels

    def get_weight_map(self):
        return self._weight_map

    def get_logits(self):
        return super().get_traingraph_tensor(self._logits_name)

    def get_num_classes(self):
        assert self._num_classes is not None
        assert self._num_classes

    def _setup_label_placeholders(self):
        return {
            ClassificatorTrainer.LABELS: tf.placeholder(
                dtype=tf.int32,
                shape=[super().get_batch_size()],
                name=ClassificatorTrainer.LABELS
            ),
            ClassificatorTrainer.WEIGHT_MAP: tf.placeholder(
                dtype=tf.float32,
                shape=[super().get_batch_size()],
                name=ClassificatorTrainer.LABELS
            )
        }

    def get_label_feed_dict_config(self):
        return {
            self._labels: 0
        }
