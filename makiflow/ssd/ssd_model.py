import tensorflow as tf
import numpy as np


class SSDModel:
    def __init__(self, dc_blocks, input_shape, num_classes):
        self.dc_blocks = dc_blocks
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        # Collecting trainable params
        self.params = []
        for dc_block in dc_blocks:
            self.params += dc_block.get_params()

        self.X = tf.placeholder(tf.float32, shape=input_shape)
        self.predictions = self.forward(self.X)
        
        # Generating default boxes
        self.default_boxes = []
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
            self.default_boxes.append(default_boxes)
            
        self.default_boxes = np.vstack(self.default_boxes)
    
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