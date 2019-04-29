class DetectorClassifierBlock():
    """
    This is the most high level part of the SSD algorith, we can call it SSD layer.
    """

    def __init__(self, layers, detector_classifier):
        self.layers = layers
        self.detector_classifier = detector_classifier

        self.params = []
        for layer in self.layers:
            self.params += layer.get_params()
        self.params += self.detector_classifier.get_params()

        # Get params and store them into python dictionary in order to save and load them correctly later
        # This data will be send to SSDModel so that it can save its weights
        self.named_params_dict = {}
        for layer in self.layers:
            self.named_params_dict.update(layer.get_params_dict())
        self.named_params_dict.update(self.detector_classifier.get_params_dict())

    def forward(self, X, is_training=False):
        """
        :return Returns list with two values: convolution out and DetectorClassifier out.
        """
        for layer in self.layers:
            X = layer.forward(X, is_training)

        return [X, self.detector_classifier.forward(X)]

    def get_dboxes(self):
        return self.detector_classifier.get_dboxes()

    def get_params(self):
        return self.params

    def to_dict(self):
        block_dict = {
            'type': 'DetectorClassifierBlock',
            'params': {
                'layers': [],
                'detector_classifier': self.detector_classifier.to_dict()
            }
        }

        for layer in self.layers:
            block_dict['params']['layers'].append(layer.to_dict())

        return block_dict

    def get_params_dict(self):
        return self.named_params_dict
