from __future__ import absolute_import
from makiflow.layers import InputLayer, ConcatLayer, ActivationLayer
from makiflow.base import MakiModel
import json
from copy import copy

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from tqdm import tqdm


class SSDModel(MakiModel):
    def __init__(self, dcs: list, input_s: InputLayer, name='MakiSSD'):
        self.dcs = dcs
        self.name = str(name)

        inputs = [input_s]
        graph_tensors = {}
        outputs = []
        for dc in dcs:
            confs, offs = dc.get_conf_offsets()
            graph_tensors.update(confs.get_previous_tensors())
            graph_tensors.update(offs.get_previous_tensors())
            graph_tensors.update(confs.get_self_pair())
            graph_tensors.update(offs.get_self_pair())

            outputs += [confs, offs]

        super().__init__(graph_tensors, outputs, inputs)
        self.input_shape = input_s.get_shape()
        self.batch_sz = self.input_shape[0]

        self._generate_default_boxes()
        self._prepare_inference_graph()
        # Get number of classes. It is needed for Focal Loss
        self._num_classes = self.dcs[0].class_number

    def _generate_default_boxes(self):
        self.default_boxes_wh = []
        # Also collect feature map sizes for later easy access to
        # particular bboxes
        self.dc_block_feature_map_sizes = []
        for dc in self.dcs:
            fmap_shape = dc.get_feature_map_shape()
            # [ batch_sz, width, height, feature_maps ]
            # We need to convert shape to int because initially it's a Dimension object
            width = int(fmap_shape[1])
            height = int(fmap_shape[2])
            self.dc_block_feature_map_sizes.append((width, height))
            dboxes = dc.get_dboxes()
            default_boxes = self.__default_box_generator(self.input_shape[1], self.input_shape[2],
                                                         width, height, dboxes)
            self.default_boxes_wh.append(default_boxes)

        self.default_boxes_wh = np.vstack(self.default_boxes_wh)

        # Converting default boxes to another format:
        # (x, y, w, h) -----> (x1, y1, x2, y2)
        self.default_boxes = copy(self.default_boxes_wh)
        # For navigation in self.default_boxes
        i = 0
        for dbox in self.default_boxes_wh:
            self.default_boxes[i] = [dbox[0] - dbox[2] / 2,  # upper left x
                                     dbox[1] - dbox[3] / 2,  # upper left y
                                     dbox[0] + dbox[2] / 2,  # bottom right x
                                     dbox[1] + dbox[3] / 2]  # bottom right y
            i += 1

        # Adjusting dboxes
        self.__correct_default_boxes(self.default_boxes)

        self.total_predictions = len(self.default_boxes)

    def get_dbox(self, dc_block_ind, dbox_category, x_pos, y_pos):
        dcblock_dboxes_to_pass = 0
        for i in range(dc_block_ind):
            dcblock_dboxes_to_pass += (
                    self.dc_block_feature_map_sizes[i][0] * self.dc_block_feature_map_sizes[i][1] *
                    len(self.dcs[i].get_dboxes())
            )
        for i in range(dbox_category):
            dcblock_dboxes_to_pass += (
                    self.dc_block_feature_map_sizes[dc_block_ind][0] * self.dc_block_feature_map_sizes[dc_block_ind][1]
            )
        dcblock_dboxes_to_pass += self.dc_block_feature_map_sizes[dc_block_ind][0] * x_pos
        dcblock_dboxes_to_pass += y_pos
        return self.default_boxes[dcblock_dboxes_to_pass]

    def __correct_default_boxes(self, dboxes):
        max_x = self.input_shape[1]
        max_y = self.input_shape[2]

        for i in range(len(dboxes)):
            # Check top left point
            dboxes[i][0] = max(0, dboxes[i][0])
            dboxes[i][1] = max(0, dboxes[i][1])
            # Check bottom right point
            dboxes[i][2] = min(max_x, dboxes[i][2])
            dboxes[i][3] = min(max_y, dboxes[i][3])

    def __default_box_generator(self, image_width, image_height, width, height, dboxes):
        """
        :param image_width - width of the input image.
        :param image_height - height of the input height.
        :param width - width of the feature map.
        :param height - height of the feature map.
        :param dboxes - list with default boxes characteristics (width, height). Example: [(1, 1), (0.5, 0.5)]
        
        :return Returns list of 4d-vectors(np.arrays) contain characteristics of the default boxes in absolute
        coordinates: center_x, center_y, height, width.
        """
        box_count = width * height
        boxes_list = []

        width_per_cell = image_width / width
        height_per_cell = image_height / height

        for w, h in dboxes:
            boxes = np.zeros((box_count, 4))

            for i in range(height):
                current_height = i * height_per_cell
                for j in range(width):
                    current_width = j * width_per_cell
                    # (x, y) coordinates of the center of the default box
                    boxes[i * width + j][0] = current_width + width_per_cell / 2  # x
                    boxes[i * width + j][1] = current_height + height_per_cell / 2  # y
                    # (w, h) width and height of the default box
                    boxes[i * width + j][2] = width_per_cell * w  # width
                    boxes[i * width + j][3] = height_per_cell * h  # height
            boxes_list.append(boxes)

        return np.vstack(boxes_list)

    def _prepare_inference_graph(self):
        confidences = []
        offsets = []

        for dc in self.dcs:
            confs, offs = dc.get_conf_offsets()
            confidences += [confs]
            offsets += [offs]

        concatenate = ConcatLayer(axis=1, name='InferencePredictionConcat' + self.name)
        self.confidences_ish = concatenate(confidences)
        self.offsets = concatenate(offsets)

        self.offsets_tensor = self.offsets.get_data_tensor()
        predicted_boxes = self.offsets_tensor + self.default_boxes

        classificator = ActivationLayer(name='Classificator' + self.name, activation=tf.nn.softmax)
        self.confidences = classificator(self.confidences_ish)
        confidences_tensor = self.confidences.get_data_tensor()

        self.predictions = [confidences_tensor, predicted_boxes]

    def predict(self, X):
        assert (self._session is not None)
        return self._session.run(
            self.predictions,
            feed_dict={self._input_data_tensors[0]: X}
        )

    def _prepare_training_graph(self):
        training_confidences = []
        training_offsets = []
        n_outs = len(self._training_outputs)
        i = 0
        while i != n_outs:
            confs, offs = self._training_outputs[i], self._training_outputs[i+1]
            training_confidences += [confs]
            training_offsets += [offs]
            i += 2

        self._train_confidences_ish = tf.concat(training_confidences, axis=1)
        self._train_offsets = tf.concat(training_offsets, axis=1)

    def _create_loss(self, loc_loss_weight, neg_samples_ratio):
        if not self._set_for_training:
            super()._setup_for_training()

        self._prepare_training_graph()
        self.input_labels = tf.placeholder(tf.int32, shape=[self.batch_sz, self.total_predictions])
        self.input_loc_loss_masks = tf.placeholder(tf.float32, shape=[self.batch_sz, self.total_predictions])
        self.input_loc = tf.placeholder(tf.float32, shape=[self.batch_sz, self.total_predictions, 4])

        confidence_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self._train_confidences_ish, labels=self.input_labels
        )
        # Calculate confidence loss for positive bboxes
        num_positives = tf.reduce_sum(self.input_loc_loss_masks)
        positive_confidence_loss = confidence_loss * self.input_loc_loss_masks
        positive_confidence_loss = tf.reduce_sum(positive_confidence_loss)
        self.positive_confidence_loss = positive_confidence_loss / num_positives

        # Calculate confidence loss for part of negative bboxes, i.e. Hard Negative Mining
        # Create binary mask for negative loss
        ones = tf.ones(shape=[self.batch_sz, self.total_predictions])
        negative_loss_mask = ones - self.input_loc_loss_masks
        negative_confidence_loss = negative_loss_mask * confidence_loss
        negative_confidence_loss = tf.reshape(
            negative_confidence_loss, shape=[self.batch_sz * self.total_predictions]
        )

        num_negatives_to_pick = tf.cast(num_positives * tf.constant(neg_samples_ratio), dtype=tf.int32)
        negative_confidence_loss, indices = tf.nn.top_k(
            negative_confidence_loss, k=num_negatives_to_pick
        )
        num_negatives_to_pick = tf.cast(num_negatives_to_pick, dtype=tf.float32)
        self.negative_confidence_loss = tf.reduce_sum(negative_confidence_loss) / num_negatives_to_pick

        final_confidence_loss = self.positive_confidence_loss + self.negative_confidence_loss

        diff = self.input_loc - self._train_offsets

        # Define smooth L1 loss
        loc_loss_l2 = 0.5 * (diff ** 2.0)
        loc_loss_l1 = tf.abs(diff) - 0.5
        smooth_l1_condition = tf.less(tf.abs(diff), 1.0)
        loc_loss = tf.where(smooth_l1_condition, loc_loss_l2, loc_loss_l1)

        loc_loss_mask = tf.stack([self.input_loc_loss_masks] * 4, axis=2)
        loc_loss = loc_loss_mask * loc_loss
        self.loc_loss = tf.reduce_sum(loc_loss) / num_positives

        loss_factor_mask_sum = tf.reduce_sum(self.input_loc_loss_masks)
        loss_factor_condition = tf.less(loss_factor_mask_sum, 1.0)
        loss_factor = tf.where(loss_factor_condition, loss_factor_mask_sum, 1.0 / loss_factor_mask_sum)

        self.loss = (final_confidence_loss + loc_loss_weight * self.loc_loss) * loss_factor

    def _create_scan_loss(self, loc_loss_weight, neg_samples_ratio):
        if not self._set_for_training:
            super()._setup_for_training()

        self._prepare_training_graph()
        self.input_labels = tf.placeholder(tf.int32, shape=[self.batch_sz, self.total_predictions])
        self.input_loc_loss_masks = tf.placeholder(tf.float32, shape=[self.batch_sz, self.total_predictions])
        self.input_loc = tf.placeholder(tf.float32, shape=[self.batch_sz, self.total_predictions, 4])

        confidence_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self._train_confidences_ish, labels=self.input_labels
        )
        print(confidence_loss.shape)
        # Calculate confidence loss for positive bboxes
        num_positives = tf.reduce_sum(self.input_loc_loss_masks)
        positive_confidence_loss = confidence_loss * self.input_loc_loss_masks
        positive_confidence_loss = tf.reduce_sum(positive_confidence_loss)
        self.positive_confidence_loss = positive_confidence_loss / num_positives
        print(self.positive_confidence_loss.shape)

        # Calculate confidence loss for part of negative bboxes, i.e. Hard Negative Mining
        # Create binary mask for negative loss
        ones = tf.ones(shape=[self.batch_sz, self.total_predictions])
        num_negatives = tf.cast(num_positives * tf.constant(neg_samples_ratio), dtype=tf.float32)
        negative_loss_mask = ones - self.input_loc_loss_masks
        negative_confidence_loss = confidence_loss * negative_loss_mask
        print(negative_confidence_loss.shape)
        num_negatives_per_batch = tf.cast(
            num_negatives / self.batch_sz,
            dtype=tf.int32
        )

        def sort_neg_losses_for_each_batch(_, batch_loss):
            top_k_negative_confidence_loss, _ = tf.nn.top_k(
                batch_loss, k=num_negatives_per_batch
            )
            return tf.reduce_sum(top_k_negative_confidence_loss)

        neg_conf_losses = tf.scan(
            fn=sort_neg_losses_for_each_batch,
            elems=negative_confidence_loss,
            infer_shape=False,
            initializer=1.0
        )
        self.negative_confidence_loss = tf.reduce_sum(neg_conf_losses) / num_negatives

        final_confidence_loss = self.positive_confidence_loss + self.negative_confidence_loss

        diff = self.input_loc - self._train_offsets

        # Define smooth L1 loss
        loc_loss_l2 = 0.5 * (diff ** 2.0)
        loc_loss_l1 = tf.abs(diff) - 0.5
        smooth_l1_condition = tf.less(tf.abs(diff), 1.0)
        loc_loss = tf.where(smooth_l1_condition, loc_loss_l2, loc_loss_l1)

        loc_loss_mask = tf.stack([self.input_loc_loss_masks] * 4, axis=2)
        loc_loss = loc_loss_mask * loc_loss
        self.loc_loss = tf.reduce_sum(loc_loss) / num_positives

        loss_factor_mask_sum = tf.reduce_sum(self.input_loc_loss_masks)
        loss_factor_condition = tf.less(loss_factor_mask_sum, 1.0)
        loss_factor = tf.where(loss_factor_condition, loss_factor_mask_sum, 1.0 / loss_factor_mask_sum)

        self.loss = (final_confidence_loss + loc_loss_weight * self.loc_loss) * loss_factor

    def _create_focal_loss(self, loc_loss_weight):
        if not self._set_for_training:
            super()._setup_for_training()

        self._prepare_training_graph()
        self.input_labels = tf.placeholder(tf.int32, shape=[self.batch_sz, self.total_predictions])
        self.input_loc_loss_masks = tf.placeholder(tf.float32, shape=[self.batch_sz, self.total_predictions])
        self.input_loc = tf.placeholder(tf.float32, shape=[self.batch_sz, self.total_predictions, 4])

        # CREATE FOCAL LOSS
        # [batch_sz, total_predictions]
        confidence_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self._train_confidences_ish, labels=self.input_labels
        )
        # [batch_sz, total_predictions, num_classes]
        train_confidences = tf.nn.softmax(self._train_confidences_ish)
        # Create one-hot encoding for picking predictions we need
        # [batch_sz, total_predictions, num_classes]
        one_hot_labels = tf.one_hot(self.input_labels, depth=self._num_classes, on_value=1.0, off_value=0.0)
        filtered_confidences = train_confidences * one_hot_labels
        # [batch_sz, total_predictions]
        sparse_confidences = tf.reduce_max(filtered_confidences, axis=-1)
        ones_arr = tf.ones(shape=[self.batch_sz, self.total_predictions], dtype=tf.float32)
        focal_weight = tf.pow(ones_arr - sparse_confidences, 1.8)
        focal_loss = tf.reduce_mean(focal_weight * confidence_loss)

        # For compability
        self.positive_confidence_loss = focal_loss
        self.negative_confidence_loss = focal_loss
        
        # CREATE LOCALIZATION LOSS
        num_positives = tf.reduce_sum(self.input_loc_loss_masks)
        # Define smooth L1 loss
        loc_loss_mask = tf.stack([self.input_loc_loss_masks] * 4, axis=2)
        self.loc_loss = tf.losses.huber_loss(
            labels=self.input_loc,
            predictions=self._train_offsets,
            weights=loc_loss_mask
        ) / num_positives

        loss_factor_condition = tf.less(num_positives, 1.0)
        loss_factor = tf.where(loss_factor_condition, num_positives, 1.0 / num_positives)
        self.loss = (focal_loss + loc_loss_weight * self.loc_loss) * loss_factor

    def fit(self, images, loc_masks, labels, gt_locs, optimizer,
            loc_loss_weight=1.0, neg_samples_ratio=3.5,
            epochs=1, loss_type='top_k_loss'):
        """
        Function for training the SSD.
        
        Parameters
        ----------
        images : numpy ndarray
            Numpy array contains images with shape [batch_sz, image_w, image_h, color_channels].
        loc_masks : numpy array
            Binary masks represent which default box matches ground truth box. In training loop it will be multiplied
            with confidence losses array in order to get only positive confidences.
        labels : numpy array
            Sparse(not one-hot encoded!) labels for classification loss. The array has a shape of [num_images].
        gt_locs : numpy ndarray
            Array with differences between ground truth boxes and default boxes coordinates: gbox - dbox.
        loc_loss_weight : float
            Means how much localization loss influences total loss:
                    loss = confidence_loss + loss_weight*localization_loss
        neg_samples_ratio : float
            Affects amount of negative samples taken for calculation confidence loss.
        optimizer : TensorFlow optimizer
            Used for minimizing loss function.
        epochs : int
            Number of epochs to run.
        loss_type : str
            Affects which loss function will be used for training.
            Options: 'scan_loss', 'top_k_loss', 'focal_loss'.
        """
        assert (optimizer is not None)
        assert (self._session is not None)
        assert (optimizer is not None)
        assert (type(loc_loss_weight) == float)
        assert (type(neg_samples_ratio) == float)

        if not self._set_for_training:
            if loss_type == 'top_k_loss':
                self._create_loss(loc_loss_weight, neg_samples_ratio)
            elif loss_type == 'scan_loss':
                self._create_scan_loss(loc_loss_weight, neg_samples_ratio)
            elif loss_type == 'focal_loss':
                self._create_focal_loss(loc_loss_weight)
            else:
                raise Exception('Unknown loss type: ' + loss_type)

        train_op = optimizer.minimize(self.loss, var_list=self._trainable_vars)
        # Initilize optimizer's variables
        self._session.run(tf.variables_initializer(optimizer.variables()))

        n_batches = len(images) // self.batch_sz

        train_loc_losses = []
        train_pos_conf_losses = []
        train_neg_conf_losses = []
        try:
            for i in range(epochs):
                print('Start shuffling...')
                images, loc_masks, labels, gt_locs = shuffle(images, loc_masks, labels, gt_locs)
                print('Finished shuffling.')
                train_loc_loss = np.float32(0)
                train_pos_conf_loss = np.float32(0)
                train_neg_conf_loss = np.float32(0)
                train_total_loss = np.float32(0)
                iterator = tqdm(range(n_batches))
                try:
                    for j in iterator:
                        img_batch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                        loc_mask_batch = loc_masks[j * self.batch_sz:(j + 1) * self.batch_sz]
                        labels_batch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                        gt_locs_batch = gt_locs[j * self.batch_sz:(j + 1) * self.batch_sz]

                        # Don't know how to fix it yet.
                        try:
                            total_loss, loc_loss_batch, pos_conf_loss_batch, neg_conf_loss_batch, _ = self._session.run(
                                [self.loss, self.loc_loss, self.positive_confidence_loss, self.negative_confidence_loss, train_op],
                                feed_dict={
                                    self._input_data_tensors[0]: img_batch,
                                    self.input_labels: labels_batch,
                                    self.input_loc_loss_masks: loc_mask_batch,
                                    self.input_loc: gt_locs_batch
                                })
                        except Exception as ex:
                            if ex is KeyboardInterrupt:
                                raise Exception('You have raised KeyboardInterrupt exception.')
                            else:
                                print(ex)
                                continue

                        # Calculate losses using exponential decay
                        train_loc_loss = 0.9 * train_loc_loss + 0.1 * loc_loss_batch
                        train_pos_conf_loss = 0.9 * train_pos_conf_loss + 0.1 * pos_conf_loss_batch
                        train_neg_conf_loss = 0.9 * train_neg_conf_loss + 0.1 * neg_conf_loss_batch
                        train_total_loss = 0.9 * train_total_loss + 0.1 * train_total_loss

                    train_loc_losses.append(train_loc_loss)
                    train_pos_conf_losses.append(train_pos_conf_loss)
                    train_neg_conf_losses.append(train_neg_conf_loss)
                    print(
                        'Epoch:', i, "Positive conf loss:", train_pos_conf_loss,
                        "Negative conf loss:", train_neg_conf_loss,
                        'Loc loss:', train_loc_loss,
                        'Total loss', train_total_loss
                    )
                except Exception as ex:
                    iterator.close()
                    print(ex)
        finally:
            return {
                'pos conf losses': train_pos_conf_losses,
                'neg conf losses': train_neg_conf_losses,
                'loc losses': train_loc_losses,
            }
