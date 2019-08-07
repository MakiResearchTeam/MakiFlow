from __future__ import absolute_import

import tensorflow as tf
from makiflow.base import MakiTensor
from makiflow.layers import ConvLayer
from makiflow.layers import ReshapeLayer


class DetectorClassifier:
    """
    This class represents a part of SSD algorithm. It consists of several parts:
    conv layers -> detector -> confidences + localization regression.
    """

    def __init__(self, reg_fms: MakiTensor, rkw, rkh, rin_f,
        class_fms: MakiTensor, ckw, ckh, cin_f, num_classes, dboxes: list, name):
        """
        Parameters
        ----------
        f_source : MakiTensor
            Source of features for the predictor. It must be feature maps from the convolutional
            network.
        kw : int
            Width of the predictor's kernel.
        kh : int
            Height of the predictor's kernel.
        in_f : int
        num_classes : int
            Number of classes to be classified + 'no object' class. 'no object' class is used for
        background.
        dboxes : list
            List of tuples of the default boxes' characteristics (width and height). Each characteristic
            must be represented in relative coordinates. Boxes' coordinates are always in the center of
            the cell of the feature map. Example:
            [(1, 1), (0.5, 1.44), (1.44, 0.5)] - (1,1) - center box matches one cell of the feature map.
        """
        self.reg_x = reg_fms
        self.rkw = rkw
        self.rkh = rkh
        self.rin_f = rin_f
        self.class_x = class_fms
        self.ckw = ckw
        self.ckh = ckh
        self.cin_f = cin_f
        self.class_number = num_classes
        self._dboxes = dboxes
        self.name = str(name)

        classifier_out_f = num_classes * len(dboxes)
        bb_regressor_out_f = 4 * len(dboxes)
        self.classifier = ConvLayer(ckw, ckh, cin_f, classifier_out_f,
                                    activation=None, padding='SAME', name='SSDClassifier_' + str(name))
        self.bb_regressor = ConvLayer(rkw, rkh, rin_f, bb_regressor_out_f,
                                      activation=None, padding='SAME', name='SSDBBDetector_' + str(name))
        self._make_detections()

    def _make_detections(self):
        """
        Creates list with "flattened" predicted confidences and regressed localization offsets for each dbox.
        Example: [confidences, offsets]
        """
        n_dboxes = len(self._dboxes)

        # FLATTEN PREDICTIONS OF THE CLASSIFIER
        confidences = self.classifier(self.class_x)
        # [BATCH SIZE, WIDTH, HEIGHT, DEPTH]
        conf_shape = confidences.get_shape()
        # width of the tensor
        conf_w = conf_shape[1]
        # height of the tensor
        conf_h = conf_shape[2]
        conf_reshape = ReshapeLayer([-1, conf_w*conf_h*n_dboxes, self.class_number], 'ConfReshape_'+self.name)
        self._confidences = conf_reshape(confidences)

        # FLATTEN PREDICTIONS OF THE REGRESSOR
        offsets = self.bb_regressor(self.reg_x)
        # [BATCH SIZE, WIDTH, HEIGHT, DEPTH]
        off_shape = offsets.get_shape()
        # width of the tensor
        off_w = off_shape[1]
        # height of the tensor
        off_h = off_shape[2]
        # 4 is for four offsets: [x1, y1, x2, y2]
        off_reshape = ReshapeLayer([-1,  off_w*off_h*n_dboxes, 4], name='OffReshape_'+self.name)
        self._offsets = off_reshape(offsets)

    def get_dboxes(self):
        return self._dboxes

    def get_conf_offsets(self):
        return [self._confidences, self._offsets]

    def get_feature_map_shape(self):
        """
        It is used for creating default boxes since their size depends
        on the feature maps' widths and heights.
        """
        return self.reg_x.get_shape()

    def to_dict(self):
        # TODO
        return {
            'type': 'DetectorClassifier',
            'params': {
                'class_number': self.class_number,
                'dboxes': self._dboxes,
                'name': self.name
            }
        }
