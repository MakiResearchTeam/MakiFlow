from __future__ import absolute_import
from makiflow.models.ssd.ssd_utils import resize_images_and_bboxes, prepare_data_v2
from makiflow.metrics.od_utils import parse_dicts
from tqdm import tqdm
import cv2
import numpy as np

"""
Helper class for fast and convenient data preparation for the SSDModel training.

Tip - use the following order to prepare your data:
1) Get annotations for your data
2) Create DataPreparator instance
3) preparator.load_images()
4) preparator.resize_images_and_bboxes((width, height))
5) preparator.normalize_images()
6) preparator.generate_masks_labels_locs(defalut_boxes)
"""


class DataPreparatorV2:
    def __init__(self, annotation_dict, name2num, path_to_data):
        """
        Parameters
        ----------
        annotation_dict : dictionary
            Contains labels, bboxes and etc for each image. (Uses the format of the dictionary XmlParser or JsonParser
            produces)
        name2num : dictionary
            Maps class names with their indices. WARNING! INDICES MUST START COUNTING FROM 1! Example:
            {
                'cat' : 1,
                'dog' : 2,
                'car' : 3
            }
        path_to_data : string
            Path to folder where images lie. Examples:
            COCO example - '/mnt/data/coco_set/train2017/'.
        """
        self._annotation_dict = annotation_dict
        self._name2num = name2num
        self._path_to_data = path_to_data

    # noinspection PyAttributeOutsideInit
    def load_images(self):
        print('Loading images, bboxes and labels...')
        self._images = []
        self._bboxes = []
        self._labels = []
        # For later usage in normalizing method
        self._images_normalized = False

        for annotation in tqdm(self._annotation_dict):
            image = cv2.imread(self._path_to_data + annotation['filename'])
            bboxes = []
            labels = []

            for gt_object in annotation['objects']:
                bboxes.append(gt_object['box'])
                labels.append(self._name2num[gt_object['name']])

            self._images.append(image)
            self._bboxes.append(bboxes)
            self._labels.append(labels)
        print('Images, bboxes and labels are loaded.')

    def __collect_image_info(self):
        self._true_boxes, self._true_labels = parse_dicts(self._annotation_dict, self._name2num)

    def resize_images_and_bboxes(self, new_size):
        """
        Resizes loaded images and bounding boxes accordingly.

        Parameters
        ----------
        new_size : tuple
            Contains new width and height. Example: (300, 300).
        """
        images, bboxes = resize_images_and_bboxes(self._images, self._bboxes, new_size)
        del self._images
        del self._bboxes
        self._images = images
        self._bboxes = bboxes
        self.__collect_image_info()

    def generate_masks_labels_locs(self, default_boxes, iou_threshold=0.5):
        """
        Generates masks, labels and locs for later usage in fit function of the SSD class.

        Parameters
        ----------
        default_boxes : array like
            List of default boxes generated by an SSDModel instance. You can get them by
            accessing `default_boxes` field in the SSDModel instance.
        iou_threshold : float
            Jaccard Index default box have to exceed to be marked as positive. Used for
            generating masks and labels.

        Returns
        -------
        masks : numpy array
            Mask vector for positive detections.
        labels : numpy array
            Vector with sparse labels of the classes. Labels are of type numpy.int32.
        locs : numpy array
            Vector contain differences in coordinates between ground truth boxes and default boxes which
            will be used for the calculation of the localization loss.
        """
        labels_arr = []
        loc_masks_arr = []
        locs_arr = []
        for true_boxes, true_labels in zip(self._true_boxes, self._true_labels):
            labels, locs, loc_mask = prepare_data_v2(true_boxes, true_labels, default_boxes, iou_threshold)
            labels_arr += [labels]
            locs_arr += [locs]
            loc_masks_arr += [loc_mask]

        self._last_labels = np.vstack(labels_arr)
        self._last_loc_masks = np.vstack(loc_masks_arr)
        self._last_locs = np.vstack(locs_arr)
        return self._last_loc_masks, self._last_labels, self._last_locs

    def get_last_masks_labels_locs(self):
        return self._last_labels, self._last_loc_masks, self._last_locs

    def normalize_images(self):
        """
        Normalizes loaded images by dividing each one by 255.

        Returns
        -------
        list
            Contains normalized images.
        """
        if self._images_normalized:
            raise Exception("Images are already normalized!")

        self._images_normalized = True
        for i in range(len(self._images)):
            self._images[i] = np.array(self._images[i], dtype=np.float32) / 255
        return self._images

    def get_images(self):
        if not self._images_normalized:
            print('Be careful, images are not normalized.')

        return self._images

    def get_bboxes(self):
        return self._bboxes

    def get_true_boxes_labels(self):
        return self._true_boxes, self._true_labels