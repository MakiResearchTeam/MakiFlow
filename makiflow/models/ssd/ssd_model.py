from __future__ import absolute_import

from makiflow.generators.gen_base import GenLayer, SSDIterator
from makiflow.layers import InputLayer, ConcatLayer, ActivationLayer
from makiflow.base import MakiModel
from makiflow.models.ssd.training_literals import TL
from makiflow.models.ssd.ssd_utils import bboxes_wh2xy, bboxes_xy2wh
from makiflow.base.loss_builder import Loss
from copy import copy

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from tqdm import tqdm


class OffsetRegression:
    DUMMY = 0
    RCNN_LIKE = 1


class SSDModel(MakiModel):
    def __init__(self, dcs: list, input_s: InputLayer, offset_reg_type=OffsetRegression.DUMMY, name='MakiSSD'):
        self.dcs = dcs
        self.name = str(name)
        self.regression_type = offset_reg_type
        inputs = [input_s]
        graph_tensors = {}
        outputs = []
        for dc in dcs:
            # `confs` shape is [batch_sz, fmap_square, num_classes]
            # `offs` shape is [batch_sz, fmap_square, 4]
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
        # For training
        self._training_vars_are_ready = False

    # -------------------------------------------------------SETTING UP DEFAULT BOXES-----------------------------------

    def _generate_default_boxes(self):
        self.dboxes_wh = []
        # Also collect feature map sizes for later easy access to
        # particular bboxes
        self.dc_block_feature_map_sizes = []
        for dc in self.dcs:
            # [batch_sz, height, width, feature_maps]
            fmap_shape = dc.get_feature_map_shape()
            height = fmap_shape[1]
            width = fmap_shape[2]
            dboxes = dc.get_dboxes()
            default_boxes = self._default_box_generator(self.input_shape[2], self.input_shape[1],
                                                        width, height, dboxes)
            self.dboxes_wh.append(default_boxes)
            self.dc_block_feature_map_sizes.append((width, height))

        self.dboxes_wh = np.vstack(self.dboxes_wh).astype(np.float32)
        # Converting default WH bboxes to XY format:
        # (x, y, w, h) -----> (x1, y1, x2, y2)
        self.dboxes_xy = bboxes_wh2xy(self.dboxes_wh)
        # Adjusting dboxes
        self._correct_dboxes()
        self.total_predictions = len(self.dboxes_xy)

    def _correct_dboxes(self):
        img_w = self.input_shape[2]
        img_h = self.input_shape[1]

        self.dboxes_xy[:, 0] = np.clip(self.dboxes_xy[:, 0], 0., img_w)  # x1
        self.dboxes_xy[:, 1] = np.clip(self.dboxes_xy[:, 1], 0., img_h)  # y1
        self.dboxes_xy[:, 2] = np.clip(self.dboxes_xy[:, 2], 0., img_w)  # x2
        self.dboxes_xy[:, 3] = np.clip(self.dboxes_xy[:, 3], 0., img_h)  # x3

        # Reassign WH dboxes to corrected values.
        self.dboxes_wh = bboxes_xy2wh(self.dboxes_xy)

    # noinspection PyMethodMayBeStatic
    def _default_box_generator(self, image_width, image_height, width, height, dboxes):
        """
        Generates default boxes in WH format.

        Parameters
        ----------
        image_width : int
            Width of the input image.
        image_height : int
            Height of the input height.
        width : int
            Width of the feature map.
        height : int
            Height of the feature map.
        dboxes : list
            List with default boxes characteristics (width, height). Example: [(1, 1), (0.5, 0.5)].

        Returns
        -------
        ndarray
            Array of 4d-vectors(np.arrays) contain characteristics of the default boxes in absolute
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

    def get_dboxes_xy(self):
        return self.dboxes_xy

    def get_dboxes_wh(self):
        return self.dboxes_wh

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
        return self.dboxes_xy[dcblock_dboxes_to_pass]

    def _get_model_info(self):
        model_dict = {
            'name': self.name,
            'input_s': self._inputs[0].get_name(),
            'reg_type': self.regression_type,
            'dcs': []
        }

        for dc in self.dcs:
            model_dict['dcs'].append(dc.to_dict())
        return model_dict

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------SETTING UP INFERENCE OF THE MODEL--------------------------

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
        # Dummy offset regression
        if self.regression_type == OffsetRegression.DUMMY:
            self.predicted_boxes = self.offsets_tensor + self.dboxes_xy
        # RCNN like offset regression
        elif self.regression_type == OffsetRegression.RCNN_LIKE:
            cx = self.dboxes_wh[:, 2] * self.offsets_tensor[:, :, 0] + self.dboxes_wh[:, 0]  # db_w * p_cx + db_cx
            cy = self.dboxes_wh[:, 3] * self.offsets_tensor[:, :, 1] + self.dboxes_wh[:, 1]  # db_h * p_cy + db_cy
            w_h = tf.exp(self.offsets_tensor[:, :, 2:])  # exponentiate width and height, magic math
            w = self.dboxes_wh[:, 2] * w_h[:, :, 0]  # times width
            h = self.dboxes_wh[:, 3] * w_h[:, :, 1]  # times height
            # [batch_sz, num_predictions] -> [batch_sz, num_predictions, 1]
            # Do this for the proper concatenation.
            cx = tf.expand_dims(cx, axis=-1)
            cy = tf.expand_dims(cy, axis=-1)
            w = tf.expand_dims(w, axis=-1)
            h = tf.expand_dims(h, axis=-1)
            predicted_bboxes_wh = tf.concat([cx, cy, w, h], axis=2)

            # Convert predicted bboxes to XY format
            up_x = predicted_bboxes_wh[:, :, 0] - predicted_bboxes_wh[:, :, 2] / 2.  # up_x
            up_y = predicted_bboxes_wh[:, :, 1] - predicted_bboxes_wh[:, :, 3] / 2.  # up_y
            bot_x = predicted_bboxes_wh[:, :, 0] + predicted_bboxes_wh[:, :, 2] / 2.  # bot_x
            bot_y = predicted_bboxes_wh[:, :, 1] + predicted_bboxes_wh[:, :, 3] / 2.  # bot_y
            up_x = tf.expand_dims(up_x, axis=-1)
            up_y = tf.expand_dims(up_y, axis=-1)
            bot_x = tf.expand_dims(bot_x, axis=-1)
            bot_y = tf.expand_dims(bot_y, axis=-1)
            self.predicted_boxes = tf.concat([up_x, up_y, bot_x, bot_y], axis=2)
        else:
            raise ValueError(f'Unknown offset regression type: {self.regression_type}')

        classificator = ActivationLayer(name='Classificator' + self.name, activation=tf.nn.softmax)
        self.confidences = classificator(self.confidences_ish)
        confidences_tensor = self.confidences.get_data_tensor()

        self.predictions = [confidences_tensor, self.predicted_boxes]

    def predict(self, X):
        assert (self._session is not None)
        return self._session.run(
            self.predictions,
            feed_dict={self._input_data_tensors[0]: X}
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------SETTING UP TRAINING-------------------------------------

    # noinspection PyAttributeOutsideInit
    def set_generator(self, generator: GenLayer):
        self._generator = generator
        if not self._set_for_training:
            super()._setup_for_training()
        if not self._training_vars_are_ready:
            self._prepare_training_vars(use_generator=True)

    def _prepare_training_vars(self, use_generator=False):
        training_confidences = []
        training_offsets = []
        n_outs = len(self._training_outputs)
        i = 0
        while i != n_outs:
            confs, offs = self._training_outputs[i], self._training_outputs[i + 1]
            training_confidences += [confs]
            training_offsets += [offs]
            i += 2

        # [batch_sz, total_predictions, num_classes]
        self._flattened_logits = tf.concat(training_confidences, axis=1)
        # [batch_sz, total_predictions, 4]
        self._train_offsets = tf.concat(training_offsets, axis=1)

        if use_generator:
            self._flattened_labels = self._generator.get_iterator()[SSDIterator.label]
            self._input_loc_loss_masks = self._generator.get_iterator()[SSDIterator.loc_mask]
            self._input_loc = self._generator.get_iterator()[SSDIterator.loc]
        else:
            # Create placeholders for the training data
            self._flattened_labels = tf.placeholder(tf.int32, shape=[self.batch_sz, self.total_predictions])
            self._input_loc_loss_masks = tf.placeholder(tf.float32, shape=[self.batch_sz, self.total_predictions])
            self._input_loc = tf.placeholder(tf.float32, shape=[self.batch_sz, self.total_predictions, 4])

        self._loc_loss_weight = tf.placeholder(tf.float32, shape=[], name='loc_loss_weight')

        # DEFINE VARIABLES NECESSARY FOR BUILDING LOSSES
        self._ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self._flattened_logits, labels=self._flattened_labels
        )
        self._num_positives = tf.reduce_sum(self._input_loc_loss_masks)
        self._build_loc_loss()
        self._training_vars_are_ready = True

        self._focal_loss_is_build = False
        self._top_k_loss_is_build = False
        self._scan_loss_is_build = False
        self._maki_loss_is_build = False
        self._quadratic_ce_loss_is_build = False

    def _build_loc_loss(self):
        diff = self._input_loc - self._train_offsets
        # Define smooth L1 loss
        loc_loss_l2 = 0.5 * (diff ** 2.0)
        loc_loss_l1 = tf.abs(diff) - 0.5
        smooth_l1_condition = tf.less(tf.abs(diff), 1.0)
        loc_loss = tf.where(smooth_l1_condition, loc_loss_l2, loc_loss_l1)

        loc_loss_mask = tf.stack([self._input_loc_loss_masks] * 4, axis=2)
        loc_loss = loc_loss_mask * loc_loss
        self._loc_loss = tf.reduce_sum(loc_loss) / self._num_positives

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------FOCAL LOSS--------------------------------------------------

    def _build_focal_loss(self):
        # [batch_sz, total_predictions, num_classes]
        train_confidences = tf.nn.softmax(self._flattened_logits)
        # Create one-hot encoding for picking predictions we need
        # [batch_sz, total_predictions, num_classes]
        one_hot_labels = tf.one_hot(self._flattened_labels, depth=self._num_classes, on_value=1.0, off_value=0.0)
        filtered_confidences = train_confidences * one_hot_labels
        # [batch_sz, total_predictions]
        sparse_confidences = tf.reduce_max(filtered_confidences, axis=-1)
        ones_arr = tf.ones(shape=[self.batch_sz, self.total_predictions], dtype=tf.float32)
        focal_weights = tf.pow(ones_arr - sparse_confidences, self._focal_gamma)
        self._focal_loss = tf.reduce_sum(focal_weights * self._ce_loss) / self._num_positives

        total_loss = self._focal_loss + self._loc_loss_weight * self._loc_loss
        condition = tf.less(self._num_positives, 1.0)
        total_loss = tf.where(condition, tf.constant(0.0), total_loss)
        self._final_focal_loss = self._build_final_loss(total_loss)
        self._focal_loss_is_build = True

    def _setup_focal_loss_inputs(self):
        self._focal_gamma = tf.placeholder(tf.float32, shape=[], name='gamma')

    def _minimize_focal_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._focal_loss_is_build:
            self._setup_focal_loss_inputs()
            self._build_focal_loss()
            self._focal_optimizer = optimizer
            self._focal_train_op = optimizer.minimize(
                self._final_focal_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._focal_optimizer != optimizer:
            print('New optimizer is used.')
            self._focal_optimizer = optimizer
            self._focal_train_op = optimizer.minimize(
                self._final_focal_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._focal_train_op

    def fit_focal(
            self, images, loc_masks, labels, gt_locs, optimizer,
            loc_loss_weight=1.0, gamma=2.0, epochs=1, global_step=None
    ):
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
        global_step : tf.Variable
            Used for learning rate exponential decay. See TensorFlow documentation on how to use
            exponential decay.
        """
        assert (type(loc_loss_weight) == float)
        assert (type(gamma) == float)

        train_op = self._minimize_focal_loss(optimizer, global_step)

        n_batches = len(images) // self.batch_sz

        iterator = None
        train_loc_losses = []
        train_focal_losses = []
        train_total_losses = []
        try:
            for i in range(epochs):
                print('Start shuffling...')
                images, loc_masks, labels, gt_locs = shuffle(images, loc_masks, labels, gt_locs)
                print('Finished shuffling.')
                loc_loss = 0
                focal_loss = 0
                total_loss = 0
                iterator = tqdm(range(n_batches))
                try:
                    for j in iterator:
                        img_batch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                        loc_mask_batch = loc_masks[j * self.batch_sz:(j + 1) * self.batch_sz]
                        labels_batch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                        gt_locs_batch = gt_locs[j * self.batch_sz:(j + 1) * self.batch_sz]

                        # Don't know how to fix it yet.
                        try:
                            batch_total_loss, batch_focal_loss, batch_loc_loss, _ = self._session.run(
                                [self._final_focal_loss, self._focal_loss, self._loc_loss, train_op],
                                feed_dict={
                                    self._input_data_tensors[0]: img_batch,
                                    self._flattened_labels: labels_batch,
                                    self._input_loc_loss_masks: loc_mask_batch,
                                    self._input_loc: gt_locs_batch,
                                    self._loc_loss_weight: loc_loss_weight,
                                    self._focal_gamma: gamma
                                })
                        except Exception as ex:
                            if ex is KeyboardInterrupt:
                                raise Exception('You have raised KeyboardInterrupt exception.')
                            else:
                                print(ex)
                                continue

                        # Calculate losses using exponential decay
                        loc_loss = 0.1 * batch_loc_loss + 0.9 * loc_loss
                        focal_loss = 0.1 * batch_focal_loss + 0.9 * focal_loss
                        total_loss = 0.1 * batch_total_loss + 0.9 * total_loss

                    train_loc_losses.append(loc_loss)
                    train_focal_losses.append(focal_loss)
                    train_total_losses.append(total_loss)
                    print(
                        'Epoch:', i,
                        'Loc loss:', loc_loss,
                        'Focal loss', focal_loss,
                        'Total loss', total_loss
                    )
                except Exception as ex:
                    iterator.close()
                    print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {
                'focal losses': train_focal_losses,
                'total losses': train_total_losses,
                'loc losses': train_loc_losses,
            }

    def genfit_focal(self, optimizer, loc_loss_weight=1.0, gamma=2.0, epochs=1, iterations=10, global_step=None):
        assert (optimizer is not None)
        assert (self._session is not None)

        train_op = self._minimize_focal_loss(optimizer, global_step)

        train_total_losses = []
        train_focal_losses = []
        train_loc_losses = []
        iterator = None
        try:
            for i in range(epochs):
                total_loss = 0
                focal_loss = 0
                loc_loss = 0
                iterator = tqdm(range(iterations))

                for _ in iterator:
                    batch_total_loss, batch_focal_loss, batch_loc_loss, _ = self._session.run(
                        [self._final_focal_loss, self._focal_loss, self._loc_loss, train_op],
                        feed_dict={
                            self._loc_loss_weight: loc_loss_weight,
                            self._focal_gamma: gamma
                        })
                    # Use exponential decay for calculating loss and error
                    focal_loss = 0.1 * batch_focal_loss + 0.9 * focal_loss
                    total_loss = 0.1 * batch_total_loss + 0.9 * total_loss
                    loc_loss = 0.1 * batch_loc_loss + 0.9 * loc_loss

                train_total_losses.append(total_loss)
                train_focal_losses.append(focal_loss)
                train_loc_losses.append(loc_loss)

                print(
                    'Epoch:', i,
                    'Focal loss: {:0.4f}'.format(float(focal_loss)),
                    'Total loss: {:0.4f}'.format(float(total_loss)),
                    'Loc loss: {:0.4f}'.format(float(loc_loss))
                )
        except Exception as ex:
            print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {
                'total loss': train_total_losses,
                'focal loss': train_focal_losses,
                'loc loss': train_loc_losses
            }

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------TOP-K LOSS----------------------------------------------

    def _build_top_k_positive_loss(self):
        positive_confidence_loss = self._ce_loss * self._input_loc_loss_masks
        positive_confidence_loss = tf.reduce_sum(positive_confidence_loss)
        self._top_k_positive_confidence_loss = positive_confidence_loss / self._num_positives

    def _build_top_k_negative_loss(self):
        # Calculate confidence loss for part of negative bboxes, i.e. Hard Negative Mining
        # Create binary mask for negative loss
        ones = tf.ones(shape=[self.batch_sz, self.total_predictions])
        negative_loss_mask = ones - self._input_loc_loss_masks
        negative_confidence_loss = negative_loss_mask * self._ce_loss
        negative_confidence_loss = tf.reshape(
            negative_confidence_loss, shape=[self.batch_sz * self.total_predictions]
        )

        num_negatives_to_pick = tf.cast(
            self._num_positives * self._top_k_neg_samples_ratio, dtype=tf.int32
        )
        negative_confidence_loss, _ = tf.nn.top_k(
            negative_confidence_loss, k=num_negatives_to_pick
        )
        num_negatives_to_pick = tf.cast(num_negatives_to_pick, dtype=tf.float32)
        self._top_k_negative_confidence_loss = tf.reduce_sum(negative_confidence_loss) / num_negatives_to_pick

    def _build_top_k_loss(self):
        # BUILDS CROSS-ENTROPY LOSS WITH BATCH HARD NEGATIVE MINING
        self._build_top_k_positive_loss()
        self._build_top_k_negative_loss()

        top_k_loss = self._top_k_positive_confidence_loss + self._top_k_negative_confidence_loss
        total_loss = top_k_loss + self._loc_loss_weight * self._loc_loss
        condition = tf.less(self._num_positives, 1.0)
        total_loss = tf.where(condition, 0.0, total_loss)
        self._final_top_k_loss = self._build_final_loss(total_loss)
        self._top_k_loss_is_build = True

    def _setup_top_k_loss_inputs(self):
        self._top_k_neg_samples_ratio = tf.placeholder(tf.float32, shape=[], name='top_k_neg_samples_ratio')

    def _minimize_top_k_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._top_k_loss_is_build:
            self._setup_top_k_loss_inputs()
            self._build_top_k_loss()
            self._top_k_optimizer = optimizer
            self._top_k_train_op = optimizer.minimize(
                self._final_top_k_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._top_k_optimizer != optimizer:
            print('New optimizer is used.')
            self._top_k_optimizer = optimizer
            self._top_k_train_op = optimizer.minimize(
                self._final_top_k_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._top_k_train_op

    def fit_top_k(
            self, images, loc_masks, labels, gt_locs, optimizer,
            loc_loss_weight=1.0, neg_samples_ratio=3.0, epochs=1, global_step=None
    ):
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
        global_step : tf.Variable
            Used for learning rate exponential decay. See TensorFlow documentation on how to use
            exponential decay.
        """
        assert (type(loc_loss_weight) == float)
        assert (type(neg_samples_ratio) == float)

        train_op = self._minimize_top_k_loss(optimizer, global_step)

        n_batches = len(images) // self.batch_sz
        iterator = None
        train_loc_losses = []
        train_neg_losses = []
        train_pos_losses = []
        train_total_losses = []
        try:
            for i in range(epochs):
                print('Start shuffling...')
                images, loc_masks, labels, gt_locs = shuffle(images, loc_masks, labels, gt_locs)
                print('Finished shuffling.')
                loc_loss = 0
                neg_loss = 0
                pos_loss = 0
                total_loss = 0
                iterator = tqdm(range(n_batches))
                try:
                    for j in iterator:
                        img_batch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                        loc_mask_batch = loc_masks[j * self.batch_sz:(j + 1) * self.batch_sz]
                        labels_batch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                        gt_locs_batch = gt_locs[j * self.batch_sz:(j + 1) * self.batch_sz]

                        # Don't know how to fix it yet.
                        try:
                            batch_total_loss, batch_p_loss, batch_n_loss, batch_loc_loss, _ = self._session.run(
                                [
                                    self._final_top_k_loss,
                                    self._top_k_positive_confidence_loss,
                                    self._top_k_negative_confidence_loss,
                                    self._loc_loss,
                                    train_op
                                ],
                                feed_dict={
                                    self._input_data_tensors[0]: img_batch,
                                    self._flattened_labels: labels_batch,
                                    self._input_loc_loss_masks: loc_mask_batch,
                                    self._input_loc: gt_locs_batch,
                                    self._loc_loss_weight: loc_loss_weight,
                                    self._top_k_neg_samples_ratio: neg_samples_ratio
                                })
                        except Exception as ex:
                            if ex is KeyboardInterrupt:
                                raise Exception('You have raised KeyboardInterrupt exception.')
                            else:
                                print(ex)
                                continue

                        # Calculate losses using exponential decay
                        loc_loss = 0.1 * batch_loc_loss + 0.9 * loc_loss
                        neg_loss = 0.1 * batch_n_loss + 0.9 * neg_loss
                        pos_loss = 0.1 * batch_p_loss + 0.9 * pos_loss
                        total_loss = 0.1 * batch_total_loss + 0.9 * total_loss

                    train_loc_losses.append(loc_loss)
                    train_neg_losses.append(neg_loss)
                    train_pos_losses.append(pos_loss)
                    train_total_losses.append(total_loss)
                    print(
                        'Epoch:', i,
                        'Loc loss:', loc_loss,
                        'Positive loss', pos_loss,
                        'Negative loss', neg_loss,
                        'Total loss', total_loss
                    )
                except Exception as ex:
                    iterator.close()
                    print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {
                'positive losses': train_pos_losses,
                'negative losses': train_neg_losses,
                'total losses': train_total_losses,
                'loc losses': train_loc_losses,
            }

    def genfit_top_k(
            self,
            optimizer,
            loc_loss_weight=1.0,
            neg_samples_ratio=3.0,
            epochs=1,
            iterations=20,
            global_step=None
    ):
        """
                Function for training the SSD.

                Parameters
                ----------
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
                global_step : tf.Variable
                    Used for learning rate exponential decay. See TensorFlow documentation on how to use
                    exponential decay.
                """
        assert (type(loc_loss_weight) == float)
        assert (type(neg_samples_ratio) == float)

        train_op = self._minimize_top_k_loss(optimizer, global_step)

        iterator = None
        train_loc_losses = []
        train_neg_losses = []
        train_pos_losses = []
        train_total_losses = []
        try:
            for i in range(epochs):
                loc_loss = 0
                neg_loss = 0
                pos_loss = 0
                total_loss = 0
                iterator = tqdm(range(iterations))
                try:
                    for j in iterator:
                        # Don't know how to fix it yet.
                        try:
                            batch_total_loss, batch_p_loss, batch_n_loss, batch_loc_loss, _ = self._session.run(
                                [
                                    self._final_top_k_loss,
                                    self._top_k_positive_confidence_loss,
                                    self._top_k_negative_confidence_loss,
                                    self._loc_loss,
                                    train_op
                                ],
                                feed_dict={
                                    self._loc_loss_weight: loc_loss_weight,
                                    self._top_k_neg_samples_ratio: neg_samples_ratio
                                })
                        except Exception as ex:
                            if ex is KeyboardInterrupt:
                                raise Exception('You have raised KeyboardInterrupt exception.')
                            else:
                                print(ex)
                                continue

                        # Calculate losses using exponential decay
                        loc_loss = 0.1 * batch_loc_loss + 0.9 * loc_loss
                        neg_loss = 0.1 * batch_n_loss + 0.9 * neg_loss
                        pos_loss = 0.1 * batch_p_loss + 0.9 * pos_loss
                        total_loss = 0.1 * batch_total_loss + 0.9 * total_loss

                    train_loc_losses.append(loc_loss)
                    train_neg_losses.append(neg_loss)
                    train_pos_losses.append(pos_loss)
                    train_total_losses.append(total_loss)
                    print(
                        'Epoch:', i,
                        'Loc loss:', loc_loss,
                        'Positive loss', pos_loss,
                        'Negative loss', neg_loss,
                        'Total loss', total_loss
                    )
                except Exception as ex:
                    iterator.close()
                    print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {
                'positive losses': train_pos_losses,
                'negative losses': train_neg_losses,
                'total losses': train_total_losses,
                'loc losses': train_loc_losses,
            }

    # ------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------SCAN LOSS----------------------------------------------

    def _build_scan_positive_loss(self):
        positive_confidence_loss = self._ce_loss * self._input_loc_loss_masks
        positive_confidence_loss = tf.reduce_sum(positive_confidence_loss)
        self._scan_positive_confidence_loss = positive_confidence_loss / self._num_positives

    def _build_scan_negative_loss(self):
        # Calculate confidence loss for part of negative bboxes, i.e. Hard Negative Mining
        # Create binary mask for negative loss
        ones = tf.ones(shape=[self.batch_sz, self.total_predictions])
        num_negatives = tf.cast(self._num_positives * self.__scan_neg_samples_ratio, dtype=tf.float32)
        negative_loss_mask = ones - self._input_loc_loss_masks
        negative_confidence_loss = self._ce_loss * negative_loss_mask
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
        self._scan_negative_confidence_loss = tf.reduce_sum(neg_conf_losses) / num_negatives

    def _build_scan_loss(self):
        # BUILDS CROSS-ENTROPY LOSS WITH PER SAMPLE HARD NEGATIVE MINING
        self._build_scan_positive_loss()
        self._build_scan_negative_loss()

        confidence_loss = self._scan_positive_confidence_loss + self._scan_negative_confidence_loss
        total_loss = confidence_loss + self._loc_loss_weight * self._loc_loss
        condition = tf.less(self._num_positives, 1.0)
        total_loss = tf.where(condition, 0.0, total_loss)
        self._final_scan_loss = self._build_final_loss(total_loss)
        self._scan_loss_is_build = True

    def _setup_scan_loss_inputs(self):
        self.__scan_neg_samples_ratio = tf.placeholder(tf.float32, shape=[], name='scan_neg_samples_ratio')

    def _minimize_scan_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._scan_loss_is_build:
            self._setup_scan_loss_inputs()
            self._build_scan_loss()
            self._scan_optimizer = optimizer
            self._scan_train_op = optimizer.minimize(
                self._final_scan_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._scan_optimizer != optimizer:
            print('New optimizer is used.')
            self._scan_optimizer = optimizer
            self._scan_train_op = optimizer.minimize(
                self._final_scan_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._scan_train_op

    def fit_scan(
            self, images, loc_masks, labels, gt_locs, optimizer,
            loc_loss_weight=1.0, neg_samples_ratio=3.0, epochs=1, global_step=None
    ):
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
        global_step : tf.Variable
            Used for learning rate exponential decay. See TensorFrow documentation on how to use
            exponential decay.
        """
        assert (type(loc_loss_weight) == float)
        assert (type(neg_samples_ratio) == float)

        train_op = self._minimize_scan_loss(optimizer, global_step)

        n_batches = len(images) // self.batch_sz

        iterator = None
        train_loc_losses = []
        train_neg_losses = []
        train_pos_losses = []
        train_total_losses = []
        try:
            for i in range(epochs):
                print('Start shuffling...')
                images, loc_masks, labels, gt_locs = shuffle(images, loc_masks, labels, gt_locs)
                print('Finished shuffling.')
                loc_loss = 0
                neg_loss = 0
                pos_loss = 0
                total_loss = 0
                iterator = tqdm(range(n_batches))
                try:
                    for j in iterator:
                        img_batch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                        loc_mask_batch = loc_masks[j * self.batch_sz:(j + 1) * self.batch_sz]
                        labels_batch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                        gt_locs_batch = gt_locs[j * self.batch_sz:(j + 1) * self.batch_sz]

                        # Don't know how to fix it yet.
                        try:
                            batch_total_loss, batch_pos_loss, batch_neg_loss, batch_loc_loss, _ = self._session.run(
                                [
                                    self._final_scan_loss,
                                    self._scan_positive_confidence_loss,
                                    self._scan_negative_confidence_loss,
                                    self._loc_loss,
                                    train_op
                                ],
                                feed_dict={
                                    self._input_data_tensors[0]: img_batch,
                                    self._flattened_labels: labels_batch,
                                    self._input_loc_loss_masks: loc_mask_batch,
                                    self._input_loc: gt_locs_batch,
                                    self._loc_loss_weight: loc_loss_weight,
                                    self.__scan_neg_samples_ratio: neg_samples_ratio
                                })
                        except Exception as ex:
                            if ex is KeyboardInterrupt:
                                raise Exception('You have raised KeyboardInterrupt exception.')
                            else:
                                print(ex)
                                continue

                        # Calculate losses using exponential decay
                        loc_loss = 0.1 * batch_loc_loss + 0.9 * loc_loss
                        neg_loss = 0.1 * batch_neg_loss + 0.9 * neg_loss
                        pos_loss = 0.1 * batch_pos_loss + 0.9 * pos_loss
                        total_loss = 0.1 * batch_total_loss + 0.9 * total_loss

                    train_loc_losses.append(loc_loss)
                    train_neg_losses.append(neg_loss)
                    train_pos_losses.append(pos_loss)
                    train_total_losses.append(total_loss)
                    print(
                        'Epoch:', i,
                        'Loc loss:', loc_loss,
                        'Positive loss', pos_loss,
                        'Negative loss', neg_loss,
                        'Total loss', total_loss
                    )
                except Exception as ex:
                    iterator.close()
                    print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {
                'positive losses': train_pos_losses,
                'negative losses': train_neg_losses,
                'total losses': train_total_losses,
                'loc losses': train_loc_losses,
            }

    def genfit_scan(
            self,
            optimizer,
            loc_loss_weight=1.0,
            neg_samples_ratio=3.0,
            epochs=1,
            iterations=20,
            global_step=None
    ):
        """
                Function for training the SSD.

                Parameters
                ----------
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
                global_step : tf.Variable
                    Used for learning rate exponential decay. See TensorFlow documentation on how to use
                    exponential decay.
                """
        assert (type(loc_loss_weight) == float)
        assert (type(neg_samples_ratio) == float)

        train_op = self._minimize_scan_loss(optimizer, global_step)

        iterator = None
        train_loc_losses = []
        train_neg_losses = []
        train_pos_losses = []
        train_total_losses = []
        try:
            for i in range(epochs):
                loc_loss = 0
                neg_loss = 0
                pos_loss = 0
                total_loss = 0
                iterator = tqdm(range(iterations))
                try:
                    for j in iterator:
                        # Don't know how to fix it yet.
                        try:
                            batch_total_loss, batch_p_loss, batch_n_loss, batch_loc_loss, _ = self._session.run(
                                [
                                    self._final_scan_loss,
                                    self._scan_positive_confidence_loss,
                                    self._scan_negative_confidence_loss,
                                    self._loc_loss,
                                    train_op
                                ],
                                feed_dict={
                                    self._loc_loss_weight: loc_loss_weight,
                                    self.__scan_neg_samples_ratio: neg_samples_ratio
                                })
                        except Exception as ex:
                            if ex is KeyboardInterrupt:
                                raise Exception('You have raised KeyboardInterrupt exception.')
                            else:
                                print(ex)
                                continue

                        # Calculate losses using exponential decay
                        loc_loss = 0.1 * batch_loc_loss + 0.9 * loc_loss
                        neg_loss = 0.1 * batch_n_loss + 0.9 * neg_loss
                        pos_loss = 0.1 * batch_p_loss + 0.9 * pos_loss
                        total_loss = 0.1 * batch_total_loss + 0.9 * total_loss

                    train_loc_losses.append(loc_loss)
                    train_neg_losses.append(neg_loss)
                    train_pos_losses.append(pos_loss)
                    train_total_losses.append(total_loss)
                    print(
                        'Epoch:', i,
                        'Loc loss:', loc_loss,
                        'Positive loss', pos_loss,
                        'Negative loss', neg_loss,
                        'Total loss', total_loss
                    )
                except Exception as ex:
                    iterator.close()
                    print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {
                'positive losses': train_pos_losses,
                'negative losses': train_neg_losses,
                'total losses': train_total_losses,
                'loc losses': train_loc_losses,
            }

    # ------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------MAKI LOSS----------------------------------------------

    def _build_maki_loss(self):
        self._maki_loss = Loss.maki_loss(
            flattened_logits=self._flattened_logits,
            flattened_labels=self._flattened_labels,
            maki_num_positives=self._num_positives,
            num_classes=self._num_classes,
            maki_gamma=self._maki_gamma,
            ce_loss=self._ce_loss
        )
        total_loss = self._loc_loss * self._loc_loss_weight + self._maki_loss
        self._final_maki_loss = self._build_final_loss(total_loss)
        self._maki_loss_is_build = True

    def _setup_maki_loss_inputs(self):
        pass

    def _minimize_maki_loss(self, optimizer, global_step, gamma):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._maki_loss_is_build:
            self._setup_maki_loss_inputs()
            self._maki_gamma = gamma
            self._build_maki_loss()
            self._maki_optimizer = optimizer
            self._maki_train_op = optimizer.minimize(
                self._final_maki_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._maki_gamma != gamma:
            print('New gamma is used.')
            self._maki_gamma = gamma
            self._build_maki_loss()
            self._maki_optimizer = optimizer
            self._maki_train_op = optimizer.minimize(
                self._final_maki_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._maki_optimizer != optimizer:
            print('New optimizer is used.')
            self._maki_optimizer = optimizer
            self._maki_train_op = optimizer.minimize(
                self._final_maki_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._maki_train_op

    def fit_maki(
            self, images, loc_masks, labels, gt_locs, optimizer,
            loc_loss_weight=1.0, gamma=2, epochs=1, global_step=None
    ):
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
        global_step : tf.Variable
            Used for learning rate exponential decay. See TensorFlow documentation on how to use
            exponential decay.
        """
        assert (type(loc_loss_weight) == float)
        assert (type(gamma) == int)

        train_op = self._minimize_maki_loss(optimizer, global_step, gamma)

        n_batches = len(images) // self.batch_sz

        iterator = None
        train_loc_losses = []
        train_maki_losses = []
        train_total_losses = []
        try:
            for i in range(epochs):
                print('Start shuffling...')
                images, loc_masks, labels, gt_locs = shuffle(images, loc_masks, labels, gt_locs)
                print('Finished shuffling.')
                loc_loss = 0
                maki_loss = 0
                total_loss = 0
                iterator = tqdm(range(n_batches))
                try:
                    for j in iterator:
                        img_batch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                        loc_mask_batch = loc_masks[j * self.batch_sz:(j + 1) * self.batch_sz]
                        labels_batch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                        gt_locs_batch = gt_locs[j * self.batch_sz:(j + 1) * self.batch_sz]

                        # Don't know how to fix it yet.
                        try:
                            batch_total_loss, batch_maki_loss, batch_loc_loss, _ = self._session.run(
                                [self._final_maki_loss, self._maki_loss, self._loc_loss, train_op],
                                feed_dict={
                                    self._input_data_tensors[0]: img_batch,
                                    self._flattened_labels: labels_batch,
                                    self._input_loc_loss_masks: loc_mask_batch,
                                    self._input_loc: gt_locs_batch,
                                    self._loc_loss_weight: loc_loss_weight,
                                })
                        except Exception as ex:
                            if ex is KeyboardInterrupt:
                                raise Exception('You have raised KeyboardInterrupt exception.')
                            else:
                                print(ex)
                                continue

                        # Calculate losses using exponential decay
                        loc_loss = 0.1 * batch_loc_loss + 0.9 * loc_loss
                        maki_loss = 0.1 * batch_maki_loss + 0.9 * maki_loss
                        total_loss = 0.1 * batch_total_loss + 0.9 * total_loss

                    train_loc_losses.append(loc_loss)
                    train_maki_losses.append(maki_loss)
                    train_total_losses.append(total_loss)
                    print(
                        'Epoch:', i,
                        'Maki loss: {:0.4f}'.format(float(maki_loss)),
                        'Total loss: {:0.4f}'.format(float(total_loss)),
                        'Loc loss: {:0.4f}'.format(float(loc_loss))
                    )
                except Exception as ex:
                    iterator.close()
                    print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {
                TL.TOTAL_LOSS: train_total_losses,
                TL.MAKI_LOSS: train_maki_losses,
                TL.LOC_LOSS: train_loc_losses
            }

    def genfit_maki(self, optimizer, loc_loss_weight=1.0, gamma=2, epochs=1, iterations=10, global_step=None):
        assert (type(gamma) is int)
        assert (optimizer is not None)
        assert (self._session is not None)

        train_op = self._minimize_maki_loss(optimizer, global_step, gamma)

        train_total_losses = []
        train_maki_losses = []
        train_loc_losses = []
        iterator = None
        try:
            for i in range(epochs):
                total_loss = 0
                maki_loss = 0
                loc_loss = 0
                iterator = tqdm(range(iterations))

                for _ in iterator:
                    batch_total_loss, batch_maki_loss, batch_loc_loss, _ = self._session.run(
                        [self._final_maki_loss, self._maki_loss, self._loc_loss, train_op],
                        feed_dict={
                            self._loc_loss_weight: loc_loss_weight
                        })
                    # Use exponential decay for calculating loss and error
                    maki_loss = 0.1 * batch_maki_loss + 0.9 * maki_loss
                    total_loss = 0.1 * batch_total_loss + 0.9 * total_loss
                    loc_loss = 0.1 * batch_loc_loss + 0.9 * loc_loss

                train_total_losses.append(total_loss)
                train_maki_losses.append(maki_loss)
                train_loc_losses.append(loc_loss)

                print(
                    'Epoch:', i,
                    'Maki loss: {:0.4f}'.format(float(maki_loss)),
                    'Total loss: {:0.4f}'.format(float(total_loss)),
                    'Loc loss: {:0.4f}'.format(float(loc_loss))
                )
        except Exception as ex:
            print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {
                TL.TOTAL_LOSS: train_total_losses,
                TL.MAKI_LOSS: train_maki_losses,
                TL.LOC_LOSS: train_loc_losses
            }

# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------QUADRATIC CE LOSS------------------------------------------

    def _build_quadratic_ce_loss(self):
        self._quadratic_ce_loss = Loss.quadratic_ce_loss(
            ce_loss=self._ce_loss
        )
        total_loss = self._loc_loss * self._loc_loss_weight + self._quadratic_ce_loss
        self._final_quadratic_ce_loss = self._build_final_loss(total_loss)
        self._quadratic_ce_loss_is_build = True

    def _setup_quadratic_ce_loss_inputs(self):
        pass

    def _minimize_quadratic_ce_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._quadratic_ce_loss_is_build:
            self._setup_quadratic_ce_loss_inputs()
            self._build_quadratic_ce_loss()
            self._quadratic_ce_optimizer = optimizer
            self._quadratic_ce_train_op = optimizer.minimize(
                self._final_quadratic_ce_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        if self._quadratic_ce_optimizer != optimizer:
            print('New optimizer is used.')
            self._quadratic_ce_optimizer = optimizer
            self._quadratic_ce_train_op = optimizer.minimize(
                self._final_quadratic_ce_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._quadratic_ce_train_op

    def fit_quadratic_ce(
            self, images, loc_masks, labels, gt_locs, optimizer,
            loc_loss_weight=1.0, epochs=1, global_step=None
    ):
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
        global_step : tf.Variable
            Used for learning rate exponential decay. See TensorFlow documentation on how to use
            exponential decay.
        """
        assert (type(loc_loss_weight) == float)

        train_op = self._minimize_quadratic_ce_loss(optimizer, global_step)

        n_batches = len(images) // self.batch_sz

        iterator = None
        train_loc_losses = []
        train_qce_losses = []
        train_total_losses = []
        try:
            for i in range(epochs):
                print('Start shuffling...')
                images, loc_masks, labels, gt_locs = shuffle(images, loc_masks, labels, gt_locs)
                print('Finished shuffling.')
                loc_loss = 0
                qce_loss = 0
                total_loss = 0
                iterator = tqdm(range(n_batches))
                try:
                    for j in iterator:
                        img_batch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                        loc_mask_batch = loc_masks[j * self.batch_sz:(j + 1) * self.batch_sz]
                        labels_batch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                        gt_locs_batch = gt_locs[j * self.batch_sz:(j + 1) * self.batch_sz]

                        # Don't know how to fix it yet.
                        try:
                            batch_total_loss, batch_maki_loss, batch_loc_loss, _ = self._session.run(
                                [self._final_maki_loss, self._maki_loss, self._loc_loss, train_op],
                                feed_dict={
                                    self._input_data_tensors[0]: img_batch,
                                    self._flattened_labels: labels_batch,
                                    self._input_loc_loss_masks: loc_mask_batch,
                                    self._input_loc: gt_locs_batch,
                                    self._loc_loss_weight: loc_loss_weight,
                                })
                        except Exception as ex:
                            if ex is KeyboardInterrupt:
                                raise Exception('You have raised KeyboardInterrupt exception.')
                            else:
                                print(ex)
                                continue

                        # Calculate losses using exponential decay
                        loc_loss = 0.1 * batch_loc_loss + 0.9 * loc_loss
                        qce_loss = 0.1 * batch_maki_loss + 0.9 * qce_loss
                        total_loss = 0.1 * batch_total_loss + 0.9 * total_loss

                    train_loc_losses.append(loc_loss)
                    train_qce_losses.append(qce_loss)
                    train_total_losses.append(total_loss)
                    print(
                        'Epoch:', i,
                        'QCE loss: {:0.4f}'.format(float(qce_loss)),
                        'Total loss: {:0.4f}'.format(float(total_loss)),
                        'Loc loss: {:0.4f}'.format(float(loc_loss))
                    )
                except Exception as ex:
                    iterator.close()
                    print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {
                TL.TOTAL_LOSS: train_total_losses,
                TL.QUADRATIC_CE_LOSS: train_qce_losses,
                TL.LOC_LOSS: train_loc_losses
            }

    def genfit_quadratic_ce(self, optimizer, loc_loss_weight=1.0, epochs=1, iterations=10, global_step=None):
        assert (optimizer is not None)
        assert (self._session is not None)

        train_op = self._minimize_quadratic_ce_loss(optimizer, global_step)

        train_total_losses = []
        train_qce_losses = []
        train_loc_losses = []
        iterator = None
        try:
            for i in range(epochs):
                total_loss = 0
                qce_loss = 0
                loc_loss = 0
                iterator = tqdm(range(iterations))

                for _ in iterator:
                    batch_total_loss, batch_qce_loss, batch_loc_loss, _ = self._session.run(
                        [self._final_quadratic_ce_loss, self._quadratic_ce_loss, self._loc_loss, train_op],
                        feed_dict={
                            self._loc_loss_weight: loc_loss_weight
                        })
                    # Use exponential decay for calculating loss and error
                    qce_loss = 0.1 * batch_qce_loss + 0.9 * qce_loss
                    total_loss = 0.1 * batch_total_loss + 0.9 * total_loss
                    loc_loss = 0.1 * batch_loc_loss + 0.9 * loc_loss

                train_total_losses.append(total_loss)
                train_qce_losses.append(qce_loss)
                train_loc_losses.append(loc_loss)

                print(
                    'Epoch:', i,
                    'QCE loss: {:0.4f}'.format(float(qce_loss)),
                    'Total loss: {:0.4f}'.format(float(total_loss)),
                    'Loc loss: {:0.4f}'.format(float(loc_loss))
                )
        except Exception as ex:
            print(ex)
        finally:
            if iterator is not None:
                iterator.close()
            return {
                TL.TOTAL_LOSS: train_total_losses,
                TL.MAKI_LOSS: train_qce_losses,
                TL.LOC_LOSS: train_loc_losses
            }
