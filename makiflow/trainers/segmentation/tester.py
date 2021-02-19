# Copyright (C) 2020  Igor Kilbas, Danil Gribanov
#
# This file is part of MakiPoseNet.
#
# MakiPoseNet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiPoseNet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

import tensorflow as tf
from pose_estimation.metrics.COCO_WholeBody import relayout_keypoints
from pose_estimation.data_preparation.coco_preparator_api import CocoPreparator
from abc import ABC, abstractmethod
import os
import skimage.io as io
from pycocotools.coco import COCO


class Tester(ABC):
    # Using in conjugation with trainer.
    # After the model was trained for some time, call the evaluation
    # method and all the info will recorded to the tensorboard.
    TEST_CONFIG = 'test_config'
    TB_FOLDER = 'tb_folder'  # folder for tensorboard to write data in
    TEST_IMAGE = 'test_image'
    BATCH_SIZE = 'batch_size'
    LOG_FOLDER = 'logs'
    ANNOT_GT_JSON = 'annot_gt_json'
    PATH_TO_VAL_IMAGES = "path_to_val_images"
    LIMIT_ANNOT = 'limit_annot'
    MIN_SIZE_H = 'min_size_h'
    MODEL_SIZE = 'model_size'
    NORMALIZATION_SHIFT = 'normalization_shift'
    NORMALIZATION_DIV = 'normalization_div'
    NORM_MODE = 'norm_mode'
    USE_BGR2RGB = 'use_bgr2rgb'
    IMG_HW = 'img_hw'
    PATH_TO_TRAIN_ANNOT = 'path_to_train_annot'
    IMAGE_IDS_FROM_TRAIN = 'image_ids_from_train'
    TEST_VIDEO = "path_to_test_video"
    TEST_VIDEO_LENGTH = 'test_video_length'
    SAVE_PREDICTED_VIDEO_FOLDER = 'folder_to_save_pred_video'

    NAME_RELAYOUR_ANNOT_JSON = "relayour_annot.json"
    NAME_PREDICTED_ANNOT_JSON = 'predicted_annot.json'
    AP_AR_DATA_TXT = 'ap_ar_data.txt'
    VIDEO_TEST = "video_test_{}.mp4"

    def __init__(self, config: dict, sess, path_to_save_logs:str):
        self._config = config[Tester.TEST_CONFIG]

        self._path_to_save_logs = os.path.join(path_to_save_logs, self.LOG_FOLDER)
        os.makedirs(self._path_to_save_logs, exist_ok=True)

        self._path_to_relayout_annot = os.path.join(self._path_to_save_logs, self.NAME_RELAYOUR_ANNOT_JSON)

        self._tb_writer = tf.summary.FileWriter(config[Tester.TB_FOLDER])
        self._sess = sess

        # Init stuff for measure metric
        self._limit_annots = self._config[self.LIMIT_ANNOT]

        self._norm_div = self._config[self.NORMALIZATION_DIV]
        self._norm_shift = self._config[self.NORMALIZATION_SHIFT]

        self._norm_mode = self._config[self.NORM_MODE]
        self._use_bgr2rgb = self._config[self.USE_BGR2RGB]
        self._model_size = self._config[self.MODEL_SIZE]
        self.min_size_h = self._config[Tester.MIN_SIZE_H]

        annot_gt = self._config[self.ANNOT_GT_JSON]

        if annot_gt is not None:
            relayout_keypoints(
                min_size_h=self.min_size_h,
                ann_file_path=self._config[self.ANNOT_GT_JSON],
                path_to_save=self._path_to_relayout_annot,
                limit_number=self._limit_annots,
            )

            # Load ground-truth annot
            self.cocoGt = COCO(self._path_to_relayout_annot)
            self._path_to_val_images = self._config[self.PATH_TO_VAL_IMAGES]
        else:
            self.cocoGt = None

        train_annot_gt = self._config.get(self.PATH_TO_TRAIN_ANNOT)

        if train_annot_gt is not None:
            self._train_annot = COCO(train_annot_gt)
            self._train_images = []
            self._ground_truth = []
            # Load each image
            ids = self._config[self.IMAGE_IDS_FROM_TRAIN]
            for single_id in ids:
                img = self._train_annot.loadImgs(single_id)[0]
                self._train_images.append(io.imread(img['coco_url']))
                # Load annotation (keypoints)
                annIds = self._train_annot.getAnnIds(imgIds=img['id'])
                anns = self._train_annot.loadAnns(annIds)
                single_ground_truth = []
                for i in range(len(anns)):
                    single_annot = anns[i]
                    # return shape (n_kp, 1, 3), slice 1 dim
                    single_ground_truth.append(CocoPreparator.take_default_skelet(single_annot)[:, 0])

                self._ground_truth.append(single_ground_truth)
        else:
            self._train_annot = None

        self._video_path = self._config[self.TEST_VIDEO]
        self._video_test_length = self._config.get(self.TEST_VIDEO_LENGTH)
        self._video_counter = 0
        self._save_pred_video_folder = self._config.get(self.SAVE_PREDICTED_VIDEO_FOLDER)

        # The summaries to write
        self._summaries = {}
        # Placeholder that take in the data for the summary
        self._summary_inputs = {}

        self._init()

    def _init(self):
        pass

    def add_image(self, name, n_images=1):
        """
        Adds an image summary to the tensorboard.
        The image dtype must by uint8 and have shape (batch_size, h, w, c).

        Parameters
        ----------
        name : str
            Name that will be displayed on the tensorboard.
        n_images : int
            Maximum number of images to display on the board.
        """
        image = tf.placeholder(dtype=tf.uint8)
        self._summary_inputs.update(
            {name: image}
        )
        image_summary = tf.summary.image(name, image, max_outputs=n_images)
        self._summaries.update(
            {name: image_summary}
        )

    def add_scalar(self, name):
        """
        Adds a scalar summary (e.g. accuracy) to the tensorboard.
        The image dtype must by float32.

        Parameters
        ----------
        name : str
            Name that will be displayed on the tensorboard.
        """
        scalar = tf.placeholder(dtype=tf.float32)
        self._summary_inputs.update(
            {name: scalar}
        )
        scalar_summary = tf.summary.scalar(name, scalar)
        self._summaries.update(
            {name: scalar_summary}
        )

    def write_summaries(self, summaries, step=None):
        """
        Writes the summary to the tensorboard log file.
        Parameters
        ----------
        summaries : dict
            Contains pairs (name, data). `data` can be whether scalar or image.
        step : int
            The training/evaluation step number.
        """
        for summary_name in summaries:
            data = summaries[summary_name]
            s_input = self._summary_inputs[summary_name]
            summary = self._summaries[summary_name]

            summary_tensor = self._sess.run(
                summary,
                feed_dict={
                    s_input: data
                }
            )
            self._tb_writer.add_summary(summary_tensor, global_step=step)
        # self._tb_writer.flush()

    @abstractmethod
    def evaluate(self, model, iteration):
        pass

    def get_writer(self):
        return self._tb_writer
