import tensorflow as tf


class SSDModel:
    def __init__(self, dc_blocks, input_shape, num_classes):
        self.dc_blocks = dc_blocks
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Collecting trainable params
        self.params = []
        for dc_block in dc_blocks:
            self.params += dc_block.get_params()
        
        self.X = tf.placeholder(tf.float32, shape=input_shape)
        self.predictions = self.forward(self.X)
        
    def set_session(self, session):
        self.session = session
        init_op = tf.variables_initializer(self.params)
        session.run(init_op)
        
    
    def forward(self, X):
        """ Returns a list of PredictionHolder objects contain information about the prediction. """
        predictions = []
        
        for dc_block in self.dc_blocks:
            dc_out = dc_block.forward(X)
            X = dc_out[0]
            predictions += dc_out[1]
        
        return predictions
    
    
    def predict(self, X):
        assert(self.session is not None)
        return self.session.run(
            self.predictions,
            feed_dict={self.X: X}
        )