class DetectorClassifierBlock():
    """ This is the most high level part of the SSD algorith, we can call it SSD layer."""
    def __init__(self, layers, detector_classifier):
        self.layers = layers
        self.detector_classifier = detector_classifier
        
        
    def forward(self, X):
        """ Returns list with two values: convolution out and DetectorClassifier out. """
        for layer in self.layers:
            X = layer.forward(out)
        
        return [X, self.detector_classifier.forward(X)]