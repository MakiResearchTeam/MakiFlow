import tensorflow as tf
import numpy as np

def jaccard_index(box_a, box_b):
    """
    Calculates Jaccard Index for two bounding boxes.
    Argumnets:
    box_a - [x_center, y_center, width, height] of the box A.
    box_a - [x_center, y_center, width, height] of the box B.
    
    Returns number represents Jaccard Index of the aforementioned bboxes.
    Example: 0.7
    
    
    Code has taken from:
    https://github.com/georgesung/ssd_tensorflow_traffic_sign_detection/blob/master/data_prep.py
    """
    pass