from __future__ import absolute_import
from makiflow.save_recover.builder import Builder
# NON-MAXIMUM SUPRESSION
from makiflow.models.ssd.ssd_utils import nms
from makiflow.tools.image_cutter import ImageCutter
import cv2
# For cutting out pieces of images with text
import numpy as np


class WordGrabber:
    # Used for the preparing input images to the SSD
    SSD_X_STEP = 150
    SSD_Y_STEP = 150


    def __init__(self, path_to_textrec, path_to_ssd):
        """
        Parameters
        ----------
        path_to_textrec : tuple
            Tuple contains both path to model's architecture and model's weights.
            Example: (path_to_json, absolute_path_to_weights).
            WARNING! PATH TO WEIGHTS MUST BE ABSOLUTE!
        path_to_ssd : tuple
            Tuple contains both path to model's architecture and model's weights.
            Example: (path_to_json, absolute_path_to_weights).
            WARNING! PATH TO WEIGHTS MUST BE ABSOLUTE!
        """
        self.text_rec_json, self.text_rec_ckpt = path_to_textrec
        self.ssd_json, self.ssd_ckpt = path_to_ssd

        # Used later. Need to initialize now.
        self.nms_conf_trashhold = 0.2
        self.nms_iou_trashhold = 0.2
    

    def load_models(self, batch_sz_ssd=1, batch_sz_text_rec=1):

        self.text_rec_model = Builder.text_recognizer_from_json(self.text_rec_json, batch_size=batch_sz_text_rec)
        self.text_rec_input_shape = self.text_rec_model.input_shape

        self.ssd_model = Builder.ssd_from_json(self.ssd_json, batch_size=batch_sz_text_rec)
        self.ssd_input_shape = self.ssd_model.input_shape
    

    def set_session(self, session):
        self.session = session

        self.text_rec_model.set_session(session)
        self.text_rec_model.load_weights(self.text_rec_ckpt)

        self.ssd_model.set_session(session)
        self.ssd_model.load_weights(self.ssd_ckpt)
    

    def set_process_params(self, nms_conf_trashhold=0.2, nms_iou_trashhold=0.2):
        # These parameters are used during Non-Maximum Supression processing
        self.nms_conf_trashhold = nms_conf_trashhold
        self.nms_iou_trashhold = nms_iou_trashhold

    
    def process(self, image):
        """
        Process given image and returns predicted text and its position (bounding box).
        Parameters
        ----------
        image : numpy ndarray
            Image to process. Image have the following shape: (image height, image width), i.e. cv2 format.
        Returns
        -------
        tuple
            Contains two lists: list with bounding boxes and list with predicted text in
            the matching bounding box.
        """
        images_to_feed, offsets = self.__prepare_image_for_ssd(image)
        predictions = self.__get_ssd_predictions(images_to_feed, offsets)
        # Perform Non-Maximum Supression
        bboxes_for_text = self.__filter_ssd_predictions(predictions)
        # Get bounded images with detected text
        images_for_text_rec = ImageCutter.get_bounded_texts(image, bboxes_for_text)
        # Prepare the images
        text_rec_input = self.__prepare_text_rec_input(images_for_text_rec)
        
        # RECOGNIZE GIVEN IMAGES
        recognized_text = []
        for image in text_rec_input:
            text = self.text_rec_model.infer_batch([image])
            recognized_text += text
        return (recognized_text, predictions[1])


    def __prepare_image_for_ssd(self, image):
        ssd_image_shape = self.ssd_input_shape[1:-1]
        # IT DOES NOT HANDLE THE CASE WHEN FEEDED IMAGE IS LESS THAN THE SSD INPUT SIZE
        pieces, offsets = ImageCutter.get_ssd_input(image, ssd_image_shape, WordGrabber.SSD_X_STEP, WordGrabber.SSD_Y_STEP)
        # Normalize each image
        for i in range(len(pieces)):
            pieces[i] = pieces[i] / 255
        return pieces, offsets

    
    def __get_ssd_predictions(self, images_to_feed, offsets):
        # Will accumulate predictions for each image
        all_confidences = []
        all_bboxes = []
        for image in images_to_feed:
            confidences, bboxes = self.ssd_model.predict([image])
            # Correct all the coordinates
            for i in range(len(bboxes)):
                bboxes[i][0] += offsets[i][0]
                bboxes[i][1] += offsets[i][1]
                bboxes[i][2] += offsets[i][0]
                bboxes[i][3] += offsets[i][1]
            
            all_confidences.append(confidences)
            all_bboxes.append(bboxes)
        return [np.vstack(all_confidences), np.vstack(all_bboxes)]
            

    def __filter_ssd_predictions(self, predictions):
        bboxes, _, _ = nms(predictions[1][0], predictions[0][0], self.nms_conf_trashhold, self.nms_iou_trashhold)
        return bboxes
    

    def __prepare_text_rec_input(self, images_for_text_rec):
        # Resize all the images to `text_rec_input_size`
        prepared_images = []
        text_rec_input_size = self.text_rec_input_shape[1:-1]
        for image in images_for_text_rec:
            new_size = (text_rec_input_size[1], text_rec_input_size[0])
            image = image[:, :, 0]
            im_shape = image.shape
            if im_shape[0] > text_rec_input_size[1] or im_shape[1] > text_rec_input_size[0]:
                image = cv2.resize(image, new_size)
            image = image.transpose([1, 0])
            im_shape = image.shape
            holder = np.ones((text_rec_input_size[0], text_rec_input_size[1], 1)) # 1 is for gray scaled image
            holder[0:im_shape[0], :im_shape[1], 0] = image
            prepared_images.append(holder)
        return prepared_images
            
            