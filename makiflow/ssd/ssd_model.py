import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from copy import copy
from tqdm import tqdm


class SSDModel:
    def __init__(self, dc_blocks, input_shape, num_classes):
        self.dc_blocks = dc_blocks
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.batch_sz = input_shape[0]
        
        # Collecting trainable params
        self.params = []
        for dc_block in dc_blocks:
            self.params += dc_block.get_params()

        
        
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
            
            
        self.total_predictions = len(self.default_boxes)
        
        # For final predicting
        confidences_ish, localization_reg = self.forward(self.X)
        confidences = tf.nn.softmax(confidences_ish)
        predicted_boxes = localization_reg + self.default_boxes
        self.predictions = [confidences, predicted_boxes]
    
    def __default_box_generator(self, image_width, image_height, width, height, dboxes):
        """
        image_width - width of the input image.
        image_height - height of the input height.
        width - width of the feature map.
        height - height of the feature map.
        dboxes - list with default boxes characteristics (width, height). Example: [(1, 1), (0.5, 0.5)]
        
        Returns list of 4d-vectors(np.arrays) contain characteristics of the default boxes in absolute coordinates:
        center_x, center_y, height, width.
        """
        box_count = width * height
        boxes_list = []
        
        width_per_cell = image_width / width
        height_per_cell = image_height / height
        
        for w, h in dboxes:
            boxes = np.zeros((box_count, 4))
            
            for i in range(height):
                current_heigth = i * height_per_cell
                for j in range(width):
                    current_width = j * width_per_cell
                    # (x, y) coordinates of the center of the default box
                    boxes[i*width + j][0] = current_width + width_per_cell/2    # x
                    boxes[i*width + j][1] = current_heigth + height_per_cell/2  # y
                    # (w, h) width and height of the default box
                    boxes[i*width + j][2] = width_per_cell * w                  # width
                    boxes[i*width + j][3] = height_per_cell * h                 # height
            boxes_list.append(boxes)
        
        return np.vstack(boxes_list)
        
        
        
        
    def set_session(self, session):
        self.session = session
        init_op = tf.variables_initializer(self.params)
        session.run(init_op)
        
    
    def forward(self, X):
        """ Returns a list of PredictionHolder objects contain information about the prediction. """
        confidences = []
        localizations = []
        
        for dc_block in self.dc_blocks:
            dcb_out = dc_block.forward(X)
            X = dcb_out[0]
            confidences.append(dcb_out[1][0])
            localizations.append(dcb_out[1][1])
            
        confidences = tf.concat(confidences, axis=1)
        localizations = tf.concat(localizations, axis=1)
        
        # Take value at 0 index because initially these tensors have
        # [bat
        return [confidences, localizations]
    
    
    def predict(self, X):
        assert(self.session is not None)
        return self.session.run(
            self.predictions,
            feed_dict={self.X: X}
        )
    
    def __smooth_L1_loss(x, alpha):
        return tf.where(tf.abs(x) < 1, x*x*0.5, 1/tf.abs(x) - 0.5)
    
    
    def fit(self, images, loc_masks, labels, gt_locs, loss_weigth=1, optimizer=None, epochs=1, test_period=1):
        assert(optimizer is not None)
        assert(self.session is not None)
        """
        images - image array for training the SSD.
        loc_masks - masks represent which default box matches ground truth box.
        labels - sparse(not one-hot encoded!) labels for classification loss.
        gt_locs - array with differences between ground truth boxes and default boxes: gbox - dbox.
        loss_weigth - means how much localization loss influences total loss:
                    loss = confidence_loss + loss_weigth*localization_loss
        """
        # Define necessary vatiables
        confidences, localizations = self.predictions
        
        input_labels = tf.placeholder(tf.int32, shape=[self.batch_sz, self.total_predictions])
        input_loc_loss_masks = tf.placeholder(tf.float32, shape=[self.batch_sz, self.total_predictions])
        input_loc = tf.placeholder(tf.float32, shape=[self.batch_sz, self.total_predictions, 4])
        
        confidence_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=confidences, labels=input_labels)
        confidence_loss = input_loc_loss_masks * confidence_loss
        confidence_loss = tf.reduce_mean(confidence_loss)
        
        diff = localizations - input_loc
        
        # Defince smooth L1 loss
        loc_loss_l2 = 0.5 * (diff**2.0)
        loc_loss_l1 = tf.abs(diff) - 0.5
        smooth_l1_condition = tf.less(tf.abs(diff), 1.0)
        loc_loss = tf.where(smooth_l1_condition, loc_loss_l2, loc_loss_l1)
        
        
        loc_loss_mask = tf.stack([input_loc_loss_masks] * 4, axis=2)
        loc_loss = loc_loss_mask * loc_loss
        loc_loss = tf.reduce_mean(loc_loss)
        
        
        loss_factor_mask_sum = tf.reduce_sum(input_loc_loss_masks)
        loss_factor_one = 1.0
        loss_factor_codition = tf.less(loss_factor_mask_sum, 1.0)
        loss_factor = tf.where(loss_factor_codition, loss_factor_one, loss_factor_mask_sum)
        
        loss = (confidence_loss + loss_weigth*loc_loss) / loss_factor
        train_op = optimizer.minimize(loss)
        # Initilize optimizer's variables
        self.session.run(tf.variables_initializer(optimizer.variables())) 
        
        n_batches = len(images) // self.batch_sz
        
        train_loc_losses = []
        train_conf_losses = []
        
        for i in range(epochs):
            images, loc_masks, labels, gt_locs = shuffle(images, loc_masks, labels, gt_locs)
            train_loc_loss = np.float32(0)
            train_conf_loss = np.float32(0)
            for j in tqdm(range(n_batches)):
                img_batch = images[j*self.batch_sz:(j+1)*self.batch_sz]
                loc_mask_batch = loc_masks[j*self.batch_sz:(j+1)*self.batch_sz]
                labels_batch = labels[j*self.batch_sz:(j+1)*self.batch_sz]
                gt_locs_batch = gt_locs[j*self.batch_sz:(j+1)*self.batch_sz]
                
                
                
                loc_loss_batch, confidence_loss_batch, _ = self.session.run(
                    [loc_loss, confidence_loss, train_op],
                    feed_dict={
                        self.X: img_batch,
                        input_labels: labels_batch,
                        input_loc_loss_masks: loc_mask_batch,
                        input_loc: gt_locs_batch
                    })
                
                
                # Calculate losses using exponetial decay
                train_loc_loss = 0.9*train_loc_loss + 0.1*loc_loss_batch
                train_conf_loss = 0.9*train_conf_loss + 0.1*confidence_loss_batch
            
            train_loc_losses.append(train_loc_loss)
            train_conf_losses.append(train_conf_loss)
            print('Epoch:', i, "Conf loss:", train_conf_loss, 'Loc loss:', train_loc_loss)
            
        return {'train loc losses': train_loc_losses,
                'train conf losses': train_conf_losses}
    