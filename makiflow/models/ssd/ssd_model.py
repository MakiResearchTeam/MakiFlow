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

        self.__generate_default_boxes()
        self.__prepare_inference_graph()
        # Get number of classes. It is needed for Focal Loss
        self.__num_classes = self.dcs[0].class_number
        # For training
        self.__training_vars_are_ready = False

#-------------------------------------------------------SETTING UP DEFAULT BOXES---------------------------------------------------------

    def __generate_default_boxes(self):
        self.default_boxes_wh = []
        # Also collect feature map sizes for later easy access to
        # particular bboxes
        self.dc_block_feature_map_sizes = []
        for dc in self.dcs:
            fmap_shape = dc.get_feature_map_shape()
            # [ batch_sz, width, height, feature_maps ]
            width = fmap_shape[1]
            height = fmap_shape[2]
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

#-------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------SETTING UP INFERENCE OF THE MODEL---------------------------------------------------------

    def __prepare_inference_graph(self):
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
    
#-------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------SETTING UP TRAINING--------------------------------------------------------------------

    def __prepare_training_graph(self):
        training_confidences = []
        training_offsets = []
        n_outs = len(self._training_outputs)
        i = 0
        while i != n_outs:
            confs, offs = self._training_outputs[i], self._training_outputs[i+1]
            training_confidences += [confs]
            training_offsets += [offs]
            i += 2

        self.__train_confidences_ish = tf.concat(training_confidences, axis=1)
        self.__train_offsets = tf.concat(training_offsets, axis=1)

        # Create placeholders for the training data
        self.__input_labels = tf.placeholder(tf.int32, shape=[self.batch_sz, self.total_predictions])
        self.__input_loc_loss_masks = tf.placeholder(tf.float32, shape=[self.batch_sz, self.total_predictions])
        self.__input_loc = tf.placeholder(tf.float32, shape=[self.batch_sz, self.total_predictions, 4])
        self.__loc_loss_weight = tf.placeholder(tf.float32, shape=[], name='loc_loss_weight')

        # DEFINE VARIABLES NECESSARY FOR BUILDING LOSSES
        self.__ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.__train_confidences_ish, labels=self.__input_labels
        )
        self.__num_positives = tf.reduce_sum(self.__input_loc_loss_masks)
        self.__training_vars_are_ready = True

        self.__focal_loss_is_build = False
        self.__top_k_loss_is_build = False
        self.__scan_loss_is_build = False
    
#-------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------FOCAL LOSS-----------------------------------------------------------------------------

    def __build_loc_loss(self):
        diff = self.__input_loc - self.__train_offsets
        # Define smooth L1 loss
        loc_loss_l2 = 0.5 * (diff ** 2.0)
        loc_loss_l1 = tf.abs(diff) - 0.5
        smooth_l1_condition = tf.less(tf.abs(diff), 1.0)
        loc_loss = tf.where(smooth_l1_condition, loc_loss_l2, loc_loss_l1)

        loc_loss_mask = tf.stack([self.__input_loc_loss_masks] * 4, axis=2)
        loc_loss = loc_loss_mask * loc_loss
        self.__loc_loss = tf.reduce_sum(loc_loss) / self.__num_positives

    def __build_focal_loss(self):
        # [batch_sz, total_predictions, num_classes]
        train_confidences = tf.nn.softmax(self.__train_confidences_ish)
        # Create one-hot encoding for picking predictions we need
        # [batch_sz, total_predictions, num_classes]
        one_hot_labels = tf.one_hot(self.__input_labels, depth=self.__num_classes, on_value=1.0, off_value=0.0)
        filtered_confidences = train_confidences * one_hot_labels
        # [batch_sz, total_predictions]
        sparse_confidences = tf.reduce_max(filtered_confidences, axis=-1)
        ones_arr = tf.ones(shape=[self.batch_sz, self.total_predictions], dtype=tf.float32)
        focal_weights = tf.pow(ones_arr - sparse_confidences, self.__gamma)
        self.__focal_loss = tf.reduce_sum(focal_weights * self.__ce_loss) / self.__num_positives

        self.__build_loc_loss()

        loss = self.__focal_loss + self.__loc_loss_weight * self.__loc_loss
        condition = tf.less(self.__num_positives, 1.0)
        self.__total_focal_loss = tf.where(condition, 0.0, loss)

        self.__focal_loss_is_build = True

    def __setup_focal_loss_inputs(self):
        self.__gamma = tf.placeholder(tf.float32, shape=[], name='gamma')

    def __minimize_focal_loss(self, optimizer):
        if not self._set_for_training:
            super()._setup_for_training()
        
        if not self.__training_vars_are_ready:
            self.__prepare_training_graph()
        
        if not self.__focal_loss_is_build:
            self.__setup_focal_loss_inputs()
            self.__build_focal_loss()
            self.__focal_optimizer = optimizer
            self.__focal_train_op = optimizer.minimize(self.__total_focal_loss, var_list=self._trainable_vars)
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self.__focal_optimizer != optimizer:
            print('New optimizer is used.')
            self.__focal_optimizer = optimizer
            self.__focal_train_op = optimizer.minimize(self.__total_focal_loss, var_list=self._trainable_vars)
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self.__focal_train_op
        
    def fit_focal(self, images, loc_masks, labels, gt_locs, optimizer,
            loc_loss_weight=1.0, gamma=2.0, epochs=1):
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
        gamma : float
            Gamma term of the focal loss. Affects how much good predictions' loss is penalized:
            more gamma - higher penalizing.
        optimizer : TensorFlow optimizer
            Used for minimizing the loss function.
        epochs : int
            Number of epochs to run.
        """
        assert (type(loc_loss_weight) == float)
        assert (type(gamma) == float)

        train_op = self.__minimize_focal_loss(optimizer)

        n_batches = len(images) // self.batch_sz

        train_loc_losses = []
        train_focal_losses = []
        train_total_losses = []
        try:
            for i in range(epochs):
                print('Start shuffling...')
                images, loc_masks, labels, gt_locs = shuffle(images, loc_masks, labels, gt_locs)
                print('Finished shuffling.')
                train_loc_loss = 0
                train_focal_loss = 0
                train_total_loss = 0
                iterator = tqdm(range(n_batches))
                try:
                    for j in iterator:
                        img_batch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                        loc_mask_batch = loc_masks[j * self.batch_sz:(j + 1) * self.batch_sz]
                        labels_batch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                        gt_locs_batch = gt_locs[j * self.batch_sz:(j + 1) * self.batch_sz]

                        # Don't know how to fix it yet.
                        try:
                            total_loss, focal_loss, loc_loss, _ = self._session.run(
                                [self.__total_focal_loss, self.__focal_loss, self.__loc_loss, train_op],
                                feed_dict={
                                    self._input_data_tensors[0]: img_batch,
                                    self.__input_labels: labels_batch,
                                    self.__input_loc_loss_masks: loc_mask_batch,
                                    self.__input_loc: gt_locs_batch,
                                    self.__loc_loss_weight: loc_loss_weight,
                                    self.__gamma: gamma
                                })
                        except Exception as ex:
                            if ex is KeyboardInterrupt:
                                raise Exception('You have raised KeyboardInterrupt exception.')
                            else:
                                print(ex)
                                continue

                        # Calculate losses using exponential decay
                        train_loc_loss +=  loc_loss
                        train_focal_loss += focal_loss
                        train_total_loss += total_loss

                    train_loc_loss /= n_batches
                    train_focal_loss /= n_batches
                    train_total_loss /= n_batches

                    train_loc_losses.append(train_loc_loss)
                    train_focal_losses.append(train_focal_loss)
                    train_total_losses.append(train_total_loss)
                    print(
                        'Epoch:', i, 
                        'Loc loss:', train_loc_loss,
                        'Focal loss', train_focal_loss,
                        'Total loss', train_total_loss
                    )
                except Exception as ex:
                    iterator.close()
                    print(ex)
        finally:
            iterator.close()
            return {
                'focal losses': train_focal_losses,
                'total losses': train_total_losses,
                'loc losses': train_loc_losses,
            }

#-------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------TOP-K LOSS-----------------------------------------------------------------------------

    def __build_top_k_positive_loss(self):
        positive_confidence_loss = self.__ce_loss * self.__input_loc_loss_masks
        positive_confidence_loss = tf.reduce_sum(positive_confidence_loss)
        self.__top_k_positive_confidence_loss = positive_confidence_loss / self.__num_positives

    def __build_top_k_negative_loss(self):
        # Calculate confidence loss for part of negative bboxes, i.e. Hard Negative Mining
        # Create binary mask for negative loss
        ones = tf.ones(shape=[self.batch_sz, self.total_predictions])
        negative_loss_mask = ones - self.__input_loc_loss_masks
        negative_confidence_loss = negative_loss_mask * self.__ce_loss
        negative_confidence_loss = tf.reshape(
            negative_confidence_loss, shape=[self.batch_sz * self.total_predictions]
        )

        num_negatives_to_pick = tf.cast(
            self.__num_positives * self.__top_k_neg_samples_ratio, dtype=tf.int32
        )
        negative_confidence_loss, _ = tf.nn.top_k(
            negative_confidence_loss, k=num_negatives_to_pick
        )
        num_negatives_to_pick = tf.cast(num_negatives_to_pick, dtype=tf.float32)
        self.__top_k_negative_confidence_loss = tf.reduce_sum(negative_confidence_loss) / num_negatives_to_pick

    def __build_top_k_loss(self):
        # BUILDS CROSS-ENTROPY LOSS WITH BATCH HARD NEGATIVE MINING
        self.__build_top_k_positive_loss()
        self.__build_top_k_negative_loss()
        self.__build_loc_loss()

        top_k_loss = self.__top_k_positive_confidence_loss + self.__top_k_negative_confidence_loss
        total_loss = top_k_loss + self.__loc_loss_weight * self.__loc_loss
        condition = tf.less(self.__num_positives, 1.0)
        self.__total_top_k_loss = tf.where(condition, 0.0, total_loss)

        self.__top_k_loss_is_build = True

    def __setup_top_k_loss_inputs(self):
        self.__top_k_neg_samples_ratio = tf.placeholder(tf.float32, shape=[], name='top_k_neg_samples_ratio')

    def __minimize_top_k_loss(self, optimizer):
        if not self._set_for_training:
            super()._setup_for_training()
        
        if not self.__training_vars_are_ready:
            self.__prepare_training_graph()
        
        if not self.__top_k_loss_is_build:
            self.__setup_top_k_loss_inputs()
            self.__build_top_k_loss()
            self.__top_k_optimizer = optimizer
            self.__top_k_train_op = optimizer.minimize(self.__total_top_k_loss, var_list=self._trainable_vars)
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self.__top_k_optimizer != optimizer:
            print('New optimizer is used.')
            self.__top_k_optimizer = optimizer
            self.__top_k_train_op = optimizer.minimize(self.__total_top_k_loss, var_list=self._trainable_vars)
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self.__top_k_train_op

    def fit_top_k(self, images, loc_masks, labels, gt_locs, optimizer,
            loc_loss_weight=1.0, neg_samples_ratio=3.0, epochs=1):
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
            Affects amount of negative samples used for calculation of the negative loss.
            Note: number of negative samples = number of positive samples * `neg_samples_ratio`
        optimizer : TensorFlow optimizer
            Used for minimizing the loss function.
        epochs : int
            Number of epochs to run.
        """
        assert (type(loc_loss_weight) == float)
        assert (type(neg_samples_ratio) == float)

        train_op = self.__minimize_top_k_loss(optimizer)

        n_batches = len(images) // self.batch_sz

        train_loc_losses = []
        train_neg_losses = []
        train_pos_losses = []
        train_total_losses = []
        try:
            for i in range(epochs):
                print('Start shuffling...')
                images, loc_masks, labels, gt_locs = shuffle(images, loc_masks, labels, gt_locs)
                print('Finished shuffling.')
                train_loc_loss = 0
                train_neg_loss = 0
                train_pos_loss = 0
                train_total_loss = 0
                iterator = tqdm(range(n_batches))
                try:
                    for j in iterator:
                        img_batch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                        loc_mask_batch = loc_masks[j * self.batch_sz:(j + 1) * self.batch_sz]
                        labels_batch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                        gt_locs_batch = gt_locs[j * self.batch_sz:(j + 1) * self.batch_sz]

                        # Don't know how to fix it yet.
                        try:
                            total_loss, p_loss, n_loss, loc_loss, _ = self._session.run(
                                [
                                    self.__total_top_k_loss,
                                    self.__top_k_positive_confidence_loss,
                                    self.__top_k_negative_confidence_loss, 
                                    self.__loc_loss,
                                    train_op
                                ],
                                feed_dict={
                                    self._input_data_tensors[0]: img_batch,
                                    self.__input_labels: labels_batch,
                                    self.__input_loc_loss_masks: loc_mask_batch,
                                    self.__input_loc: gt_locs_batch,
                                    self.__loc_loss_weight: loc_loss_weight,
                                    self.__top_k_neg_samples_ratio: neg_samples_ratio
                                })
                        except Exception as ex:
                            if ex is KeyboardInterrupt:
                                raise Exception('You have raised KeyboardInterrupt exception.')
                            else:
                                print(ex)
                                continue

                        # Calculate losses using exponential decay
                        train_loc_loss +=  loc_loss
                        train_neg_loss += n_loss
                        train_pos_loss += p_loss
                        train_total_loss += total_loss

                    train_loc_loss /= n_batches
                    train_neg_loss /= n_batches
                    train_pos_loss /= n_batches
                    train_total_loss /= n_batches

                    train_loc_losses.append(train_loc_loss)
                    train_neg_losses.append(train_neg_loss)
                    train_pos_losses.append(train_pos_loss)
                    train_total_losses.append(train_total_loss)
                    print(
                        'Epoch:', i, 
                        'Loc loss:', train_loc_loss,
                        'Positive loss', train_pos_loss,
                        'Negative loss', train_neg_loss,
                        'Total loss', train_total_loss
                    )
                except Exception as ex:
                    iterator.close()
                    print(ex)
        finally:
            iterator.close()
            return {
                'positive losses': train_pos_losses,
                'negative losses': train_neg_losses,
                'total losses': train_total_losses,
                'loc losses': train_loc_losses,
            }
    
#-------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------SCAN LOSS-----------------------------------------------------------------------------

    def __build_scan_positive_loss(self):
        positive_confidence_loss = self.__ce_loss * self.__input_loc_loss_masks
        positive_confidence_loss = tf.reduce_sum(positive_confidence_loss)
        self.__scan_positive_confidence_loss = positive_confidence_loss / self.__num_positives

    def __build_scan_negative_loss(self):
        # Calculate confidence loss for part of negative bboxes, i.e. Hard Negative Mining
        # Create binary mask for negative loss
        ones = tf.ones(shape=[self.batch_sz, self.total_predictions])
        num_negatives = tf.cast(self.__num_positives * self.__scan_neg_samples_ratio, dtype=tf.float32)
        negative_loss_mask = ones - self.__input_loc_loss_masks
        negative_confidence_loss = self.__ce_loss * negative_loss_mask
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
        self.__scan_negative_confidence_loss = tf.reduce_sum(neg_conf_losses) / num_negatives

    def __build_scan_loss(self):
        # BUILDS CROSS-ENTROPY LOSS WITH PER SAMPLE HARD NEGATIVE MINING
        self.__build_scan_positive_loss()
        self.__build_scan_negative_loss()
        self.__build_loc_loss()

        confidence_loss = self.__scan_positive_confidence_loss + self.__scan_negative_confidence_loss
        total_loss = confidence_loss + self.__loc_loss_weight * self.__loc_loss
        condition = tf.less(self.__num_positives, 1.0)
        self.__total_scan_loss = tf.where(condition, 0.0, total_loss)
        self.__scan_loss_is_build = True

    def __setup_scan_loss_inputs(self):
        self.__scan_neg_samples_ratio = tf.placeholder(tf.float32, shape=[], name='scan_neg_samples_ratio')

    def __minimize_scan_loss(self, optimizer):
        if not self._set_for_training:
            super()._setup_for_training()
        
        if not self.__training_vars_are_ready:
            self.__prepare_training_graph()
        
        if not self.__scan_loss_is_build:
            self.__setup_scan_loss_inputs()
            self.__build_scan_loss()
            self.__scan_optimizer = optimizer
            self.__scan_train_op = optimizer.minimize(self.__total_scan_loss, var_list=self._trainable_vars)
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self.__scan_optimizer != optimizer:
            print('New optimizer is used.')
            self.__scan_optimizer = optimizer
            self.__scan_train_op = optimizer.minimize(self.__total_scan_loss, var_list=self._trainable_vars)
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self.__scan_train_op

    def fit_scan(self, images, loc_masks, labels, gt_locs, optimizer,
            loc_loss_weight=1.0, neg_samples_ratio=3.0, epochs=1):
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
            Affects amount of negative samples used for calculation of the negative loss.
            Note: number of negative samples = number of positive samples * `neg_samples_ratio`
        optimizer : TensorFlow optimizer
            Used for minimizing the loss function.
        epochs : int
            Number of epochs to run.
        """
        assert (type(loc_loss_weight) == float)
        assert (type(neg_samples_ratio) == float)

        train_op = self.__minimize_scan_loss(optimizer)

        n_batches = len(images) // self.batch_sz

        train_loc_losses = []
        train_neg_losses = []
        train_pos_losses = []
        train_total_losses = []
        try:
            for i in range(epochs):
                print('Start shuffling...')
                images, loc_masks, labels, gt_locs = shuffle(images, loc_masks, labels, gt_locs)
                print('Finished shuffling.')
                train_loc_loss = 0
                train_neg_loss = 0
                train_pos_loss = 0
                train_total_loss = 0
                iterator = tqdm(range(n_batches))
                try:
                    for j in iterator:
                        img_batch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                        loc_mask_batch = loc_masks[j * self.batch_sz:(j + 1) * self.batch_sz]
                        labels_batch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                        gt_locs_batch = gt_locs[j * self.batch_sz:(j + 1) * self.batch_sz]

                        # Don't know how to fix it yet.
                        try:
                            total_loss, p_loss, n_loss, loc_loss, _ = self._session.run(
                                [
                                    self.__total_scan_loss,
                                    self.__scan_positive_confidence_loss,
                                    self.__scan_negative_confidence_loss, 
                                    self.__loc_loss,
                                    train_op
                                ],
                                feed_dict={
                                    self._input_data_tensors[0]: img_batch,
                                    self.__input_labels: labels_batch,
                                    self.__input_loc_loss_masks: loc_mask_batch,
                                    self.__input_loc: gt_locs_batch,
                                    self.__loc_loss_weight: loc_loss_weight,
                                    self.__scan_neg_samples_ratio: neg_samples_ratio
                                })
                        except Exception as ex:
                            if ex is KeyboardInterrupt:
                                raise Exception('You have raised KeyboardInterrupt exception.')
                            else:
                                print(ex)
                                continue

                        # Calculate losses using exponential decay
                        train_loc_loss +=  loc_loss
                        train_neg_loss += n_loss
                        train_pos_loss += p_loss
                        train_total_loss += total_loss

                    train_loc_loss /= n_batches
                    train_neg_loss /= n_batches
                    train_pos_loss /= n_batches
                    train_total_loss /= n_batches

                    train_loc_losses.append(train_loc_loss)
                    train_neg_losses.append(train_neg_loss)
                    train_pos_losses.append(train_pos_loss)
                    train_total_losses.append(train_total_loss)
                    print(
                        'Epoch:', i, 
                        'Loc loss:', train_loc_loss,
                        'Positive loss', train_pos_loss,
                        'Negative loss', train_neg_loss,
                        'Total loss', train_total_loss
                    )
                except Exception as ex:
                    iterator.close()
                    print(ex)
        finally:
            iterator.close()
            return {
                'positive losses': train_pos_losses,
                'negative losses': train_neg_losses,
                'total losses': train_total_losses,
                'loc losses': train_loc_losses,
            }

    