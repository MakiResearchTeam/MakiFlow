from __future__ import absolute_import
from makiflow.base import MakiModel, MakiTensor
from makiflow.layers import InputLayer
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from makiflow.models.classificator.utils import error_rate, sparse_cross_entropy
from copy import copy

EPSILON = np.float32(1e-37)


class ClassificatorBasis(MakiModel):
    def __init__(self, input: InputLayer, output: MakiTensor, name='MakiClassificator'):
        graph_tensors = copy(output.get_previous_tensors())
        # Add output tensor to `graph_tensors` since it doesn't have it.
        # It is assumed that graph_tensors contains ALL THE TENSORS graph consists of.
        graph_tensors.update(output.get_self_pair())
        outputs = [output]
        inputs = [input]
        super().__init__(graph_tensors, outputs, inputs)
        self.name = str(name)
        self._batch_sz = input.get_shape()[0]
        self._images = self._input_data_tensors[0]
        self._inference_out = self._output_data_tensors[0]
        self._softmax_out = tf.nn.softmax(self._inference_out)
        # For training
        self._training_vars_are_ready = False
        # Identity transformation
        self._labels_transform = lambda x: x
        self._labels = None

    def _get_model_info(self):
        input_mt = self._inputs[0]
        output_mt = self._outputs[0]
        return {
            'input_mt': input_mt.get_name(),
            'output_mt': output_mt.get_name(),
            'name': self.name
        }

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------SETTING UP TRAINING-------------------------------------

    def set_label_input(self, labels, labels_transform):
        """
        Replaces basic labels placeholder with the custom one.
        Parameters
        ----------
        labels : tf.placeholder
            The placeholder for the input data.
        labels_transform : python function
            This transformation will be applied to the labels placeholder for later loss calculation.
        """
        if labels is None:
            raise ValueError(f'Please provide the necessary tf.placeholder. Got {labels}')
        if labels_transform is None:
            raise ValueError(f'Please provide the necessary transformation function. Got {labels_transform}')
        self._labels_transform = labels_transform
        self._labels = labels

    def _prepare_training_vars(self):
        if not self._set_for_training:
            super()._setup_for_training()

        self._logits = self._training_outputs[0]
        self._num_classes = self._logits.get_shape()[-1]
        # Number of labels does not always equal to the size of the batch size (with RNNs).
        # It is more safe to take the first dimension as the number of the labels.
        num_labels = self._logits.get_shape()[0]
        if self._labels is None:
            self._labels = tf.placeholder(tf.int32, shape=[num_labels])
        self._ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self._logits, labels=self._labels_transform(self._labels)
        )
        self._training_vars_are_ready = True

    def evaluate(self, Xtest, Ytest):
        Xtest = Xtest.astype(np.float32)
        Yish_test = tf.nn.softmax(self._inference_out)
        n_batches = Xtest.shape[0] // self._batch_sz

        test_cost = 0
        predictions = np.zeros(len(Xtest))
        for k in tqdm(range(n_batches)):
            Xtestbatch = Xtest[k * self._batch_sz:(k + 1) * self._batch_sz]
            Ytestbatch = Ytest[k * self._batch_sz:(k + 1) * self._batch_sz]
            Yish_test_done = self._session.run(Yish_test, feed_dict={self._images: Xtestbatch}) + EPSILON
            test_cost += sparse_cross_entropy(Yish_test_done, Ytestbatch)
            predictions[k * self._batch_sz:(k + 1) * self._batch_sz] = np.argmax(Yish_test_done, axis=1)

        error_r = error_rate(predictions, Ytest)
        test_cost = test_cost / (len(Xtest) // self._batch_sz)
        return error_r, test_cost

    def predict(self, Xtest, use_softmax=True):
        if use_softmax:
            out = self._softmax_out
        else:
            out = self._inference_out
        n_batches = len(Xtest) // self._batch_sz

        predictions = []
        for i in tqdm(range(n_batches)):
            Xbatch = Xtest[i * self._batch_sz:(i + 1) * self._batch_sz]
            predictions += [self._session.run(out, feed_dict={self._images: Xbatch})]

        return np.vstack(predictions, axis=0)

