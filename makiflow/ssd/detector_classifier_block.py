class DetectorClassifierBlock():
    """ This is the most high level part of the SSD algorith, we can call it SSD layer."""
    def __init__(self, layers, detector_classifier):
        self.layers = layers
        self.detector_classifier = detector_classifier
        
        self.params = []
        for layer in self.layers:
            self.params += layer.get_params()
        self.params += self.detector_classifier.get_params()
        
        
    def forward(self, X):
        """ Returns list with two values: convolution out and DetectorClassifier out. """
        for layer in self.layers:
            X = layer.forward(X)
        
        return [X, self.detector_classifier.forward(X)]
    
    
    def get_dboxes(self):
        return self.detector_classifier.get_dboxes()
    
    
    def get_params(self):
        return self.params