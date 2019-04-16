from __future__ import absolute_import

from makiflow.layers import ConvLayer, FlattenLayer
import tensorflow as tf

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
        self.name = str(name)
        self.class_number = class_number
        self.dboxes = dboxes
        self.classifier_out_f = class_number * len(dboxes)
        self.detector_out_f = 4 * len(dboxes)
        self.dboxes_count = len(dboxes)
        
        self.classifier = ConvLayer(kw, kh, in_f, self.classifier_out_f, 
                                    activation=None, name='Classifier'+str(name))
        self.detector = ConvLayer(kw, kh, in_f, self.detector_out_f,
                                  activation=None, name='Detector'+str(name))
        
    
    def get_dboxes(self):
        return self.dboxes
    
    
    def forward(self, X):
        """ Returns list with "flattened" predicted confidences and regressed localization offsets for each dbox.
        Example: [confidences np_array, offsets np_array]"""
        confidences = self.classifier.forward(X)
        confidences_w = confidences.shape[1]    # width
        confidences_h = confidences.shape[2]    # height
        confidences = tf.reshape(confidences,[-1, 
                                              confidences_w*confidences_h*self.dboxes_count, self.class_number])
        
        offsets = self.detector.forward(X)
        offsets_w = offsets.shape[1]            # width
        offsets_h = offsets.shape[2]            # height
        offsets = tf.reshape(offsets, [-1, 
                                       offsets_w*offsets_h*self.dboxes_count, 4]) # 4 is for four offsets: [x, y, width, height]
        
        return [confidences, offsets]
    
    
    def get_params(self):
        """ Returns trainable params of the DetectorClassifier"""
        return [*self.classifier.get_params(), *self.detector.get_params()]
    
    
    def get_params_dict(self):
        return self.core.get_params_dict()
        
    
        