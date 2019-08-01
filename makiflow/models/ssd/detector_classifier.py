from __future__ import absolute_import

import tensorflow as tf
from makiflow.base import MakiTensor
from makiflow.layers import ConvLayer


class DetectorClassifier:
    """
    This class represents a part of SSD algorithm. It consists of several parts:
    conv layers -> detector -> confidences + localization regression.
    """

    def __init__(self, f_source: MakiTensor, kw, kh, in_f, class_number, dboxes, name):
        """
        input_shape - list represents shape of the tensor which is input for the DetectorClassifier. Example:
        [8, 8, 128] - 8, 8 - spacial dimensions, 128 - number of feature maps.
        :param class_number - number of classes to be classified + 'no object' class. 'no object' class is used for
        effective training and making more correct predictions.
        :param dboxes - list of tuples of the default boxes' characteristics (width and height). Each characteristic must be represented
        in relative coordinates. Boxes' coordinates are always in the center of the cell of the feature map. Example: [(1, 1), (0.5, 1.44), (1.44, 0.5)] - (1,1) - center box matches one cell of the feature map. 
        """
        self.x = f_source
        self.kw = kw
        self.kh = kh
        self.in_f = in_f
        self.class_number = class_number
        self.dboxes = dboxes
        self.name = str(name)

        classifier_out_f = class_number * len(dboxes)
        bb_regressor_out_f = 4 * len(dboxes)
        self.classifier = ConvLayer(kw, kh, in_f, classifier_out_f,
                                    activation=None, name='SSDClassifier' + str(name))
        self.bb_regressor = ConvLayer(kw, kh, in_f, bb_regressor_out_f,
                                      activation=None, name='SSDBBDetector' + str(name))
        self._make_detections()

    def get_dboxes(self):
        return self.dboxes

    def _make_detections(self):
        """
        :return Returns list with "flattened" predicted confidences and regressed localization offsets for each dbox.
        Example: [confidences np_array, offsets np_array]
        """
        X = self.x
        n_dboxes = len(self.dboxes)

        # FLATTEN PREDICTIONS OF THE CLASSIFIER
        confidences = self.classifier(X)
        # [BATCH SIZE, WIDTH, HEIGHT, DEPTH]
        conf_shape = confidences.get_shape()
        # width of the tensor
        conf_w = conf_shape[1]
        # height of the tensor
        conf_h = conf_shape[2]
        confidences = tf.reshape(confidences, [-1,
                                              conf_w*conf_h*n_dboxes, self.class_number])

        # FLATTEN PREDICTIONS OF THE REGRESSOR
        offsets = self.bb_regressor(X)
        off_shape = offsets.get_shape()
        off_w = off_shape[1]            # width of the tensor
        off_h = off_shape[2]            # height of the tensor
        offsets = tf.reshape(offsets, [-1, 
                                       off_w*off_h*n_dboxes,
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
                'f_source_name': self.x.get_name(),
                'kw': self.kw,
                'kh': self.kh,
                'in_f': self.in_f,
                'class_number': self.class_number,
                'dboxes': self.dboxes,
                'name': self.name
            }
        }
