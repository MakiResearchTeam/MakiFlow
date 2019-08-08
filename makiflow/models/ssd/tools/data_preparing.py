from __future__ import absolute_import
from makiflow.models.ssd.ssd_utils import resize_images_and_bboxes, prepare_data
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
class DataPreparator:
    def __init__(self, annotation_dict, class_name_to_num, path_to_data):
        """
        Parameters
        ----------
        annotations : dictionary
            Contains labels, bboxes and etc for each image. (Uses the format of the dictionary XmlParser or JsonParser
            produces)
        class_name_to_num : dictionary
            Maps class names with their indices. WARNING! INDICES MUST START COUNTING FROM 1! Example:
            {
                'cat' : 1,
                'dog' : 2,
                'car' : 3
            }
        path_to_data : string
            Path to folder where images lie. Examples:
            COCO example - '/mnt/data/coco_set/train2017/'.
        num_files : int
            Number of annotations (and images) to load later. Leave it None if you want to load all the data.
        """
        self.__annotation_dict = annotation_dict
        self.__class_name_to_num = class_name_to_num
        self.__path_to_data = path_to_data
        
        
    
    def load_images(self):
        print('Loading images, bboxes and labels...')
        self.__images = []
        self.__bboxes = []
        self.__labels = []
        # For later usage in normalizing method
        self.__images_normalized = False
        
        for annotation in tqdm(self.__annotation_dict):
            image = cv2.imread(self.__path_to_data + annotation['filename'])
            bboxes = []
            labels = []
            
            for gt_object in annotation['objects']:
                bboxes.append(gt_object['box'])
                labels.append( self.__class_name_to_num[gt_object['name']] )
                
            self.__images.append(image)
            self.__bboxes.append(bboxes)
            self.__labels.append(labels)
        print('Images, bboxes and labels are loaded.')
        
        
    def __collect_image_info(self):
        self.__images_info = []  # Used in prepare_data function
        for labels, bboxes in zip(self.__labels, self.__bboxes):
            image_info = {
                'bboxes': bboxes,
                'classes': labels
            }
            self.__images_info.append(image_info)
        
        
    def resize_images_and_bboxes(self, new_size):
        """ 
        Resizes loaded images and bounding boxes accordingly.
        
        Parameters
        ----------
        new_size : tuple
            Contains new width and height. Example: (300, 300).
        """
        resize_images_and_bboxes(self.__images, self.__bboxes, new_size)
        self.__collect_image_info()
    
    
    def generate_masks_labels_locs(self, default_boxes, iou_trashhold=0.5):
        """
        Generates masks, labels and locs for later usage in fit function of the SSD class.
        
        Parameters
        ----------
        default_boxes : array like
            List of default boxes generated by an SSDModel instance. You can get them by
            accessing `default_boxes` field in the SSDModel instance.
        iou_trashhold : float
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
        labels = []
        loc_masks = []
        gt_locs = []
        for image_info in tqdm(self.__images_info):
            prepared_data = prepare_data(image_info, default_boxes, iou_trashhold)
            labels.append(prepared_data['labels'])
            loc_masks.append(prepared_data['loc_mask'])
            gt_locs.append(prepared_data['gt_locs'])
        
        self.__last_labels = np.array(labels, dtype=np.int32)
        self.__last_loc_masks = np.array(loc_masks, dtype=np.float32)
        self.__last_gt_locs = np.array(gt_locs, dtype=np.float32)
        return self.__last_loc_masks, self.__last_labels, self.__last_gt_locs
    
    
    def get_last_masks_labels_locs(self):
        return self.__last_labels, self.__last_loc_masks, self.__last_gt_locs
    
    
    def normalize_images(self):
        """
        Normalizes loaded images by dividing each one by 255.
        
        Returns
        -------
        list
            Contains normalized images.
        """
        if self.__images_normalized:
            raise Exception("Images are already normalized!")
            
        self.__images_normalized = True
        for i in range(len(self.__images)):
            self.__images[i] = np.array(self.__images[i], dtype=np.float32) / 255
        return self.__images
    
    
    def get_images(self):
        if not self.__images_normalized:
            print('Be careful, images are not normalized.')
        
        return self.__images

    def get_bboxes(self):
        return self.__bboxes

    def get_images_info(self):
        return self.__images_info

        
        
        
        
        
        
        
        
        
        
        
        
            
            
        