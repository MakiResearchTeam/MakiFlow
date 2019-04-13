class SSDModel:
    def __init__(self, dc_blocks, input_shape, num_classes):
        self.dc_blocks = dc_blocks
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    
    def forward(self, X):
        """ Returns a list of PredictionHolder objects contain information about the prediction. """
        predictions = []
        
        for dc_block in self.dc_blocks:
            dc_out = dc_block.forward(X)
            X = dc_out[0]
            predictions += dc_out[1]
        
        return predictions