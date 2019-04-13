from __future__ import absolute_import

from makiflow.layers import ConvLayer


class PredictionHolder:
    """ Helper class for organizing values predicted by SSD. It holds confidences and localization regression values for
    a particular type of default box."""
    def __init__(self, conf, loc, dbox):
        """
        conf - tensor with confidences for the dbox.
        loc - regressed value for localization e.g. default box correction.
        dbox - tuple with default box characteristics (width, height). Example: (1, 1).
        """
        self.conv = conf
        self.loc = loc
        self.dbox = dbox

class DetectorClassifier():
    """ This class represents a part of SSD algorithm. It consists of several parts:
    conv layers -> detector -> confidences + localization regression.
    """
    def __init__(self, kw, kh, in_f, class_number, dboxes, name):
        """
        input_shape - list represents shape of the tensor which is input for the DetectorClassifier. Example:
        [8, 8, 128] - 8, 8 - spacial dimensions, 128 - number of feature maps.
        class_number - number of classes to be classified + 'no object' class. 'no object' class is used for
        effective training and making more correct predictions.
        dboxes - list of tuples of the default boxes' characteristics (width and height). Each characteristic must be represented 
        in relative coordinates. Boxes' coordinates are always in the center of the cell of the feature map. Example: [(1, 1), (0.5, 1.44), (1.44, 0.5)] - (1,1) - center box matches one cell of the feature map. 
        """
        self.name = name
        self.class_number = class_number
        self.dboxes = dboxes
        out_f = len(dboxes)*(class_number+4)
        self.core = ConvLayer(kw, kh, in_f, out_f, activation=None, name='DetectorClassifier'=str(name))
    
    
    def forward(self, X):
        """ Returns a list of PredictionHolder objects contain information about the prediction. """
        conv_out = self.core.forward(X)
        step = class_number+4
        predictions = []
        for i in range(len(self.dboxes)):
            conf_loc = conv_out[:, :, :, i*step: (i+1)*step]
            conf = conf_loc[:, :, :, :class_number]
            loc = conf_loc[:, :, :, class_number:]
            predictions.append(PredictionHolder(conv, loc, self.dboxes[i]))
        return predictions
    
    
    def get_params(self):
        """ Returns trainable params of the DetectorClassifier"""
        return self.core.get_params()
    
    
    def get_params_dict(self):
        return self.core.get_params_dict()
        
    
        