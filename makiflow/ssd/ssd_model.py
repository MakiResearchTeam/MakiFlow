from copy import copy

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tqdm import tqdm

# For saving the architecture
import json


class SSDModel:
    def __init__(self, dc_blocks, input_shape, num_classes, name='MakiSSD'):
        self.dc_blocks = dc_blocks
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.batch_sz = input_shape[0]
        self.name = str(name)
        self.session = None

        # Collecting trainable params
        self.params = []
        for dc_block in dc_blocks:
            self.params += dc_block.get_params()

        # Get params and store them into python dictionary in order to save and load them correctly later
        self.named_params_dict = {}
        for dc_block in self.dc_blocks:
            self.named_params_dict.update(dc_block.get_params_dict())

        self.X = tf.placeholder(tf.float32, shape=input_shape)
        # Generating default boxes
        self.default_boxes_wh = []
        out = self.X
        for dc_block in dc_blocks:
            out = dc_block.forward(out)
            # out = [ convolution_output, detector_classifier_output ]
            out = out[0]
            # out[0].shape = [ batch_sz, width, height, feature_maps ]
            # We need to convert shape to int because initially it's a Dimension object
            width = int(out.shape[1])
            height = int(out.shape[2])
            dboxes = dc_block.get_dboxes()
            default_boxes = self.__default_box_generator(input_shape[1], input_shape[2],
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

        # For final predicting
        confidences_ish, localization_reg = self.forward(self.X)
        confidences = tf.nn.softmax(confidences_ish)
        predicted_boxes = localization_reg + self.default_boxes
        self.predictions = [confidences, predicted_boxes]

        
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

    
    def set_session(self, session):
        self.session = session
        init_op = tf.variables_initializer(self.params)
        session.run(init_op)

        
    def save_weights(self, path):
        """
        This function uses default TensorFlow's way for saving models - checkpoint files.
        :param path - full path+name of the model.
        Example: '/home/student401/my_model/model.ckpt'
        """
        assert (self.session is not None)
        saver = tf.train.Saver(self.named_params_dict)
        save_path = saver.save(self.session, path)
        print('Model saved to %s' % save_path)

        
    def load_weights(self, path):
        """
        This function uses default TensorFlow's way for restoring models - checkpoint files.
        :param path - full path+name of the model.
        Example: '/home/student401/my_model/model.ckpt'
        """
        assert (self.session is not None)
        saver = tf.train.Saver(self.named_params_dict)
        saver.restore(self.session, path)
        print('Model restored')

        
    def to_json(self, path):
        """
        Convert model's architecture to json file and save it.
        path - path to file to save in.
        """

        model_dict = {
            'name': self.name,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
        }

        dc_blocks_dict = {
            'dc_blocks': []
        }
        for dc_block in self.dc_blocks:
            dc_blocks_dict['dc_blocks'].append(dc_block.to_dict())

        model_dict.update(dc_blocks_dict)
        model_json = json.dumps(model_dict, indent=1)
        json_file = open(path, mode='w')
        json_file.write(model_json)
        json_file.close()
        print("Model's architecture is saved to {}.".format(path))

        
    def forward(self, X, is_training=False):
        """
        Returns a list of PredictionHolder objects contain information about the prediction.
        """
        confidences = []
        localizations = []

        for dc_block in self.dc_blocks:
            dcb_out = dc_block.forward(X, is_training)
            X = dcb_out[0]
            confidences.append(dcb_out[1][0])
            localizations.append(dcb_out[1][1])

        confidences = tf.concat(confidences, axis=1)
        localizations = tf.concat(localizations, axis=1)

        # Take value at 0 index because initially these tensors have
        # [bat
        return [confidences, localizations]

    
    def predict(self, X):
        assert (self.session is not None)
        return self.session.run(
            self.predictions,
            feed_dict={self.X: X}
        )

    
    def fit(self, images, loc_masks, labels, gt_locs, loc_loss_weigth=1, neg_samples_ration=3.5, optimizer=None, epochs=1, test_period=1):
        """
        Function for training the SSD.
        
        :param images - array for training the SSD.
        :param loc_masks - masks represent which default box matches ground truth box.
        :param labels - sparse(not one-hot encoded!) labels for classification loss.
        :param gt_locs - array with differences between ground truth boxes and default boxes: gbox - dbox.
        :param loss_weigth - means how much localization loss influences total loss:
                    loss = confidence_loss + loss_weight*localization_loss
        :param neg_samples_ratio - affects amount of negative samples taken for calculation confidence loss.
        """
        assert(optimizer is not None)
        assert(self.session is not None)
        
        # TODO test_period - не используется
        assert (optimizer is not None)
        assert (self.session is not None)
        # Define necessary vatiables
        confidences, localizations = self.forward(self.X, is_training=True)
        input_labels = tf.placeholder(tf.int32, shape=[self.batch_sz, self.total_predictions])
        input_loc_loss_masks = tf.placeholder(tf.float32, shape=[self.batch_sz, self.total_predictions])
        input_loc = tf.placeholder(tf.float32, shape=[self.batch_sz, self.total_predictions, 4])

        confidence_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=confidences, labels=input_labels)
        # Calculate confidence loss for positive bboxes
        positive_confidence_loss = input_loc_loss_masks * confidence_loss
        positive_confidence_loss = tf.reduce_mean(confidence_loss)
        # Calculate confidence loss for part of negative bboxes e.g. Hard Negative Mining
        num_positives = tf.reduce_sum(input_loc_loss_masks)
        negative_confidence_loss = tf.reshape(confidence_loss, shape=[self.batch_sz*self.total_predictions])
        negative_confidence_loss, indices = tf.nn.top_k(negative_confidence_loss, k=tf.cast(num_positives*tf.constant(neg_samples_ration), dtype=tf.int32))
        negative_confidence_loss = tf.reduce_sum(negative_confidence_loss) / np.float32(self.total_predictions)

        final_confidence_loss = positive_confidence_loss + negative_confidence_loss

        diff = input_loc - localizations

        # Defince smooth L1 loss
        loc_loss_l2 = 0.5 * (diff ** 2.0)
        loc_loss_l1 = tf.abs(diff) - 0.5
        smooth_l1_condition = tf.less(tf.abs(diff), 1.0)
        loc_loss = tf.where(smooth_l1_condition, loc_loss_l2, loc_loss_l1)

        loc_loss_mask = tf.stack([input_loc_loss_masks] * 4, axis=2)
        loc_loss = loc_loss_mask * loc_loss
        loc_loss = tf.reduce_mean(loc_loss)

        loss_factor_mask_sum = tf.reduce_sum(input_loc_loss_masks)
        loss_factor_condition = tf.less(loss_factor_mask_sum, 1.0)
        loss_factor = tf.where(loss_factor_condition, loss_factor_mask_sum, 1.0 / loss_factor_mask_sum)

        loss = (final_confidence_loss + loc_loss_weigth * loc_loss) * loss_factor
        train_op = optimizer.minimize(loss)
        # Initilize optimizer's variables
        self.session.run(tf.variables_initializer(optimizer.variables()))

        n_batches = len(images) // self.batch_sz

        train_loc_losses = []
        train_conf_losses = []

        for i in range(epochs):
            print('Start shuffling...')
            images, loc_masks, labels, gt_locs = shuffle(images, loc_masks, labels, gt_locs)
            print('Finished shuffling.')
            train_loc_loss = np.float32(0)
            train_conf_loss = np.float32(0)
            for j in tqdm(range(n_batches)):
                img_batch = images[j * self.batch_sz:(j + 1) * self.batch_sz]
                loc_mask_batch = loc_masks[j * self.batch_sz:(j + 1) * self.batch_sz]
                labels_batch = labels[j * self.batch_sz:(j + 1) * self.batch_sz]
                gt_locs_batch = gt_locs[j * self.batch_sz:(j + 1) * self.batch_sz]

                # Don't know how to fix it yet.
                try:
                    loc_loss_batch, confidence_loss_batch, _ = self.session.run(
                        [loc_loss, final_confidence_loss, train_op],
                        feed_dict={
                            self.X: img_batch,
                            input_labels: labels_batch,
                            input_loc_loss_masks: loc_mask_batch,
                            input_loc: gt_locs_batch
                        })
                except:
                    continue

                # Calculate losses using exponetial decay
                train_loc_loss = 0.9 * train_loc_loss + 0.1 * loc_loss_batch
                train_conf_loss = 0.9 * train_conf_loss + 0.1 * confidence_loss_batch

            train_loc_losses.append(train_loc_loss)
            train_conf_losses.append(train_conf_loss)
            print('Epoch:', i, "Conf loss:", train_conf_loss, 'Loc loss:', train_loc_loss)

        return {'train loc losses': train_loc_losses,
                'train conf losses': train_conf_losses}
