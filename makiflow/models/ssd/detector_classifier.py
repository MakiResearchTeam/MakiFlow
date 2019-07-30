from __future__ import absolute_import

import tensorflow as tf

from makiflow.layers import ConvLayer


class DetectorClassifier:
    """
    This class represents a part of SSD algorithm. It consists of several parts:
    conv layers -> detector -> confidences + localization regression.
    """

    def __init__(self, f_source, kw, kh, in_f, class_number, dboxes, name):
        """
        input_shape - list represents shape of the tensor which is input for the DetectorClassifier. Example:
        [8, 8, 128] - 8, 8 - spacial dimensions, 128 - number of feature maps.
        :param class_number - number of classes to be classified + 'no object' class. 'no object' class is used for
        effective training and making more correct predictions.
        :param dboxes - list of tuples of the default boxes' characteristics (width and height). Each characteristic must be represented
        in relative coordinates. Boxes' coordinates are always in the center of the cell of the feature map. Example: [(1, 1), (0.5, 1.44), (1.44, 0.5)] - (1,1) - center box matches one cell of the feature map. 
        """
        self.name = str(name)
        self.class_number = class_number
        self.dboxes = dboxes
        self.dboxes_count = len(dboxes)
        self.x = f_source

        # Collect info for saving
        self.classifier_shape = self.classifier.shape
        self.detector_shape = self.bb_regressor.shape

        self.classifier_out_f = class_number * len(dboxes)
        self.detector_out_f = 4 * len(dboxes)
        self.classifier = ConvLayer(kw, kh, in_f, self.classifier_out_f,
                                    activation=None, name='Classifier' + str(name))
        self.bb_regressor = ConvLayer(kw, kh, in_f, self.detector_out_f,
                                      activation=None, name='Detector' + str(name))

        # Collect params and store them into python dictionary in order save and load correctly in the future
        self.named_params_dict = {}
        self.named_params_dict.update(self.classifier.get_params_dict())
        self.named_params_dict.update(self.bb_regressor.get_params_dict())

    def get_dboxes(self):
        return self.dboxes

    def _make_detections(self):
        """
        :return Returns list with "flattened" predicted confidences and regressed localization offsets for each dbox.
        Example: [confidences np_array, offsets np_array]
        """
        X = self.x
        confidences = self.classifier(X)

        conf_shape = confidences.get_shape()
        confidences_w = conf_shape[1]    # width of the tensor
        confidences_h = conf_shape[2]    # height of the tensor
        confidences = tf.reshape(confidences, [-1,
                                              confidences_w*confidences_h*self.dboxes_count, self.class_number])
        
        offsets = self.bb_regressor(X)
        offsets_shape = offsets.get_shape()
        offsets_w = offsets_shape[1]            # width of the tensor
        offsets_h = offsets_shape[2]            # height of the tensor
        offsets = tf.reshape(offsets, [-1, 
                                       offsets_w*offsets_h*self.dboxes_count,
                                       4])      # 4 is for four offsets: [x1, y1, x2, y2]
        return [confidences, offsets]

    def get_params(self):
        """
        :return Returns trainable params of the DetectorClassifier
        """
        return [*self.classifier.get_params(), *self.bb_regressor.get_params()]

    def get_params_dict(self):
        return self.named_params_dict

    def to_dict(self):
        return {
            'type': 'DetectorClassifier',
            'params': {
                'name': self.name,
                'class_number': self.class_number,
                'dboxes': self.dboxes,
                'classifier_shape': list(self.classifier_shape),
                'detector_shape': list(self.detector_shape)
            }
        }
