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
from scipy.special import binom
from .utils import _process_labels


class Loss:
    MAKI_LOSS = 'MAKI_LOSS'
    FOCAL_LOSS = 'FOCAL_LOSS'
    CE_LOSS = 'CE_LOSS'
    QCE_LOSS = 'QCE_LOSS'
    POLY_LOSS = 'POLY_LOSS'

    @staticmethod
    def dice_loss(p, g, eps, axes=None):
        """
        Computes Dice loss according to the formula from:
        V-Net: Fully Convolutional Neural Networks forVolumetric Medical Image Segmentation
        Link to the paper: http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf

        Parameters
        ----------
        p : tf.Tensor
            Predicted probabilities.
        g : tf.Tensor
            Ground truth labels.
        eps : float
            Used to prevent division by zero in the Dice denominator.
        axes : list
            Defines which axes the dice value will be computed on. The computed dice values will be averaged
            along the remaining axes. If None, Dice is computed on an entire batch.

        Returns
        -------
        tf.Tensor
            Scalar dice loss tensor.
        """

        numerator = p * g
        # [batch_size, -1]
        numerator = tf.layers.flatten(numerator)
        # [batch_size]
        numerator = tf.reduce_sum(numerator, axis=axes)

        p_squared = tf.square(p)
        p_squared = tf.layers.flatten(p_squared)
        p_squared = tf.reduce_sum(p_squared, axis=axes)
        # g is not squared to avoid unnecessary computation.
        # 0^2 = 0
        # 1^2 = 1
        g_squared = tf.reduce_sum(g, axis=axes)
        denominator = p_squared + g + eps

        dice = 2 * numerator / denominator
        return 1 - tf.reduce_mean(dice)

    @staticmethod
    def maki_loss(
            flattened_logits,
            flattened_labels,
            num_positives,
            num_classes,
            maki_gamma,
            ce_loss
    ):
        """
        Creates Maki Loss using polynomials.
        Parameters
        ----------
        flattened_logits : tf.Tensor
            Tensor of flattened logits with shape [batch_sz, total_predictions, num_classes].
        flattened_labels : tf.Tensor
            Tensor of flattened labels with shape [batch_sz, total_predictions].
        num_positives: tf.Tensor
            Tensor of shape [batch_sz]
        num_classes : int
            Number of classes.
        maki_gamma : int
            The Maki Loss hyperparameter. Higher the `maki_gamma` - higher the penalty for the
            predominant classes.
        ce_loss : tf.Tensor
            Tensor with the cross-entropy loss of shape [batch_sz, total_predictions].
        Returns
        -------
        tf.Tensor
            Constructed Maki loss.
        """
        # [batch_sz, total_predictions, num_classes]
        train_confidences = tf.nn.softmax(flattened_logits)
        # Create one-hot encoding for picking predictions we need
        # [batch_sz, total_predictions, num_classes]
        one_hot_labels = tf.one_hot(flattened_labels, depth=num_classes, on_value=1.0, off_value=0.0)
        filtered_confidences = train_confidences * one_hot_labels
        # [batch_sz, total_predictions]
        sparse_confidences = tf.reduce_max(filtered_confidences, axis=-1)
        # Create Maki polynomial
        maki_polynomial = tf.constant(0.0)
        for k in range(1, maki_gamma + 1):
            # Do subtraction because gradient must be with minus as well
            # Maki loss grad: -(1 - p)^gamma / p
            # CE loss grad: - 1 / p
            maki_polynomial -= Loss._create_maki_polynomial_part(k, sparse_confidences, maki_gamma) - \
                               Loss._create_maki_polynomial_part(k, tf.ones_like(sparse_confidences), maki_gamma)

        num_positives = tf.reduce_sum(num_positives)
        return tf.reduce_sum(maki_polynomial + ce_loss) / num_positives

    @staticmethod
    def _create_maki_polynomial_part(k, sparse_confidences, maki_gamma):
        binomial_coeff = binom(maki_gamma, k)
        powered_p = tf.pow(-1.0 * sparse_confidences, k)
        return binomial_coeff * powered_p / (1.0 * k)

    @staticmethod
    def focal_loss(
            logits,
            labels,
            num_classes,
            focal_gamma=2.0,
            num_positives=None,
            ce_loss=None,
            raw_tensor=False
    ):
        """
        Creates Focal Loss.
        Parameters
        ----------
        logits : tf.Tensor
            Tensor of flattened logits with shape [batch_sz, total_predictions, num_classes].
        labels : tf.Tensor
            Tensor of flattened labels with shape [batch_sz, total_predictions].
        num_positives : tf.Tensor
            Tensor of shape [batch_sz], contains number of hard examples per sample. If the `num_positives` is
            provided, then the loss will be divided by the sum of `num_positives`.
        num_classes : int
            Number of classes.
        focal_gamma : float
            The Focal Loss hyperparameter. Higher the `focal_gamma` - higher the penalty for the
            predominant classes.
        ce_loss : tf.Tensor
            Tensor with the cross-entropy loss of shape [batch_sz, total_predictions].
        raw_tensor : bool
            Whether to return a sum of all values or an original loss tensor with shape [batch_sz, ...].
        Returns
        -------
        tf.Tensor
            Constructed Focal loss.
        """
        if ce_loss is None:
            ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        # [batch_sz, total_predictions, num_classes]
        train_confidences = tf.nn.softmax(logits)
        # Create one-hot encoding for picking predictions we need
        # [batch_sz, total_predictions, num_classes]
        one_hot_labels = tf.one_hot(labels, depth=num_classes, on_value=1.0, off_value=0.0)
        filtered_confidences = train_confidences * one_hot_labels
        # [batch_sz, total_predictions]
        sparse_confidences = tf.reduce_max(filtered_confidences, axis=-1)
        ones_arr = tf.ones_like(labels, dtype=tf.float32)
        focal_weights = tf.pow(ones_arr - sparse_confidences, focal_gamma)
        focal_loss = focal_weights * ce_loss

        if num_positives is not None:
            num_positives = tf.reduce_sum(num_positives)
            focal_loss = focal_loss / num_positives

        if raw_tensor:
            return focal_loss

        return tf.reduce_sum(focal_loss)

    @staticmethod
    def focal_binary_loss(
            logits,
            labels,
            focal_gamma=2.0,
            num_positives=None,
            ce_loss=None,
            raw_tensor=False,
            label_smoothing=None
    ):
        """
        Creates Focal Loss.
        Parameters
        ----------
        logits : tf.Tensor
            Tensor of flattened logits with shape [batch_sz, total_predictions, num_classes].
        labels : tf.Tensor
            Tensor of flattened labels with shape [batch_sz, total_predictions, num_classes].
        num_positives : tf.Tensor
            Tensor of shape [batch_sz], contains number of hard examples per sample. If the `num_positives` is
            provided, then the loss will be divided by the sum of `num_positives`.
        focal_gamma : float
            The Focal Loss hyperparameter. Higher the `focal_gamma` - higher the penalty for the
            predominant classes.
        ce_loss : tf.Tensor
            Tensor with the cross-entropy loss of shape [batch_sz, total_predictions].
        raw_tensor : bool
            Whether to return a sum of all values or an original loss tensor with shape [batch_sz, ...].
        Returns
        -------
        tf.Tensor
            Constructed Focal loss.
        """
        if label_smoothing is not None:
            labels = _process_labels(labels, label_smoothing)
            train_confidences = tf.sigmoid(logits)
            # Terms for the positive and negative class components of the loss
            pos_term = labels * ((1 - train_confidences) ** focal_gamma)
            neg_term = (1 - labels) * (train_confidences ** focal_gamma)
            # Term involving the log and ReLU
            log_weight = pos_term
            log_weight += neg_term
            log_term = tf.math.log1p(tf.math.exp(-tf.math.abs(logits)))
            log_term += tf.nn.relu(-logits)
            log_term *= log_weight

            # Combine all the terms into the loss
            focal_loss = neg_term * logits + log_term
        else:
            if ce_loss is None:
                ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
            # [batch_sz, total_predictions, num_classes]
            train_confidences = tf.sigmoid(logits)
            # [batch_sz, total_predictions, num_classes]
            loss_pos = tf.pow(1 - train_confidences, focal_gamma)
            loss_neg = tf.pow(train_confidences, focal_gamma)
            label_mask = tf.cast(labels, tf.bool)
            # True  - y = 1, (1-p)**gamma * log(p)
            # False - y = 0, p ** gamma * log(1 - p)
            loss_mask = tf.where(label_mask, loss_pos, loss_neg)
            focal_loss = loss_mask * ce_loss

        if num_positives is not None:
            num_positives = tf.reduce_sum(num_positives)
            focal_loss = focal_loss / num_positives

        if raw_tensor:
            return focal_loss

        return tf.reduce_sum(focal_loss)

    @staticmethod
    def quadratic_ce_loss(ce_loss, num_positives=None):
        """
        Creates QuadraticCE Loss from pure CE Loss.
        Parameters
        ----------
        ce_loss : tf.Tensor
            Tensor with the cross-entropy loss of shape [batch_sz, total_predictions].
        num_positives : tf.Tensor
            Tensor of shape [batch_sz], contains number of hard examples per sample.
            If `num_positives` set to None the QCE loss will be normalized by taking the mean
            over a batch.
        Returns
        -------
        tf.Tensor
            Constructed QuadraticCE loss.
        """
        quadratic_ce = ce_loss * ce_loss / 2.0
        if num_positives is not None:
            return tf.reduce_sum(quadratic_ce) / tf.reduce_sum(num_positives)
        return tf.reduce_mean(quadratic_ce)

    @staticmethod
    def poly_loss(flattened_logits, flattened_labels, num_classes, num_positives, coeffs):
        """
        Creates Poly Loss from the polynomial coefficients.
        Parameters
        ----------
        flattened_logits : tf.Tensor
            Tensor of flattened logits with shape [batch_sz, total_predictions, num_classes].
        flattened_labels : tf.Tensor
            Tensor of flattened labels with shape [batch_sz, total_predictions].
        num_classes : int
            Number of classes.
        num_positives : tf.Tensor
            Tensor of shape [batch_sz]. Contains number of positive samples for each training sample.
        coeffs : list
            List of the polynomial coefficients.
        Returns
        -------
        tf.Tensor
            Constructed Poly loss.
        """
        # [batch_sz, total_predictions, num_classes]
        train_confidences = tf.nn.softmax(flattened_logits)
        # Create one-hot encoding for picking predictions we need
        # [batch_sz, total_predictions, num_classes]
        one_hot_labels = tf.one_hot(flattened_labels, depth=num_classes, on_value=1.0, off_value=0.0)
        filtered_confidences = train_confidences * one_hot_labels
        # [batch_sz, total_predictions]
        sparse_confidences = tf.reduce_max(filtered_confidences, axis=-1)
        loss = 0.0
        for i, c in enumerate(coeffs):
            loss += tf.reduce_sum(tf.pow(sparse_confidences, i)) * c
        num_positives = tf.reduce_sum(num_positives)
        return loss / num_positives

    @staticmethod
    def abs_loss(labels, predictions, raw_tensor=False):
        diff = tf.abs(labels - predictions)
        if raw_tensor:
            return diff
        return tf.reduce_mean(diff)

    @staticmethod
    def mse_loss(labels, predictions, raw_tensor=False):
        diff = tf.square(labels - predictions)
        if raw_tensor:
            return diff
        return tf.reduce_mean(diff)
