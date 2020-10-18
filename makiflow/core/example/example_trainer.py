from __future__ import absolute_import
from makiflow.core.training import MakiTrainer, Loss
import tensorflow as tf
from ..utils import to_makitensor


class ExampleTrainer(MakiTrainer):
    def get_label_feed_dict_config(self):
        return {
            to_makitensor(self._labels, 'labels'): 0
        }

    def _setup_for_training(self):
        # Always call the super()._setup_for_training() first
        super()._setup_for_training()
        # Define here all the necessary variables
        self._labels = tf.placeholder(dtype='float32', shape=[None], name='labels')

    def _build_loss(self):
        # This method must return a scalar of the training loss
        out_xs = self.get_model().get_outputs()
        losses = []
        for out_x in out_xs:
            out_t = out_x.get_data_tensor()
            losses += [Loss.mse_loss(self._labels, out_t)]

        return tf.add_n(losses)
