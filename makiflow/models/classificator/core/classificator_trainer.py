from makiflow.core import MakiTrainer, MakiModel
import tensorflow as tf
from abc import ABC


class ClassificatorTrainer(MakiTrainer, ABC):
    LABELS = 'LABELS'

    def __init__(self, model: MakiModel, train_inputs: list, num_classes: int = None, label_tensors: dict = None):
        """
        Parameters
        ----------
        model : MakiModel
            The model's object.
        train_inputs : list
            List of the input training MakiTensors. Their names must be the same as their inference counterparts!
        num_classes : int
            The number of classes the model will be trained on.
        label_tensors : dict
            Contains pairs (tensor_name, tf.Tensor), where tf.Tensor contains the required training data.

        """
        self._num_classes = num_classes
        super().__init__(model, train_inputs, label_tensors)

    def _setup_for_training(self):
        super()._setup_for_training()
        self._labels = super().get_label_tensors()[ClassificatorTrainer.LABELS]
        logits_makitensor = super().get_model().get_logits()
        self._logits_training_tensor = super().get_traingraph_tensor(logits_makitensor.get_name())

    def get_labels(self):
        return self._labels

    def get_logits(self):
        return self._logits_training_tensor

    def get_num_classes(self):
        assert self._num_classes is not None, 'Number of classes is not provided. Please provide the number of ' \
                                              'classes in the constructor since the trainer requires this information.'
        assert self._num_classes

    def _setup_label_placeholders(self):
        return {
            ClassificatorTrainer.LABELS: tf.placeholder(dtype=tf.int32, shape=[super().get_batch_size()])
        }

    def get_label_feed_dict_config(self):
        return {
            self._labels: 0
        }
