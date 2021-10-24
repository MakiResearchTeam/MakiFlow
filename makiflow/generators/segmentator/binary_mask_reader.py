import numpy as np
import cv2
from glob import glob
from os.path import join


# This class was made off a gen wrapper with the same name
# TODO: use this class in the corresponding wrapper
class BinaryMaskLoader:
    def __init__(self, n_classes: int, image_shape: tuple, class_priority=None):
        """
        Reads binary masks from the mask folder and aggregates them into a label tensor.
        By default all the masks are being aggregated into a tensor of labels of
        shape (height, width, n_classes).
        If `class_priority` is provided, all the masks are aggregated into a tensor of labels
        of shape (height, width, 1). Pixel value is an index of a class with highest priority that is present
        at that location of the image.
        This wrapper is caching all the masks and images, so be careful when using it on large datasets.

        Parameters
        ----------
        n_classes : int
            Number of classes in the data.
        image_shape : tuple
            (Heights, Width).
        class_priority : arraylike, optional
            Used to merge binary masks into a single-channel multiclass mask.
            First comes class with the highest priority.

        Notes
        -----
        The data must be organized as follows:
        - /images
            - 1.bmp
            - 2.bmp
            - ...
        - /masks
            - /1 - names of an image and its corresponding mask folder must be the same.
                - 1.bmp - name of the mask in this case is the class id.
                - 3.bmp
            - /2
                - 1.bmp
                - 5.bmp
                - 6.bmp
            - ...
        """
        self.n_classes = n_classes
        self.image_shape = image_shape
        self.class_priority = class_priority

    def load_mask(self, folder_path):
        # Load individual binary masks into a tensor of masks
        label_tensor = np.zeros(shape=(*self.image_shape, self.n_classes), dtype='int32')
        for binary_mask_path in glob(join(folder_path, '*')):
            filename = binary_mask_path.split('/')[-1]
            class_id = int(filename.split('.')[0])
            assert class_id != 0, 'Encountered class 0. Since 0 is reserved for the background, ' \
                                  'class names (filenames) must start from 1.'
            binary_mask = cv2.imread(binary_mask_path)
            assert binary_mask is not None, f'Could not load mask with path={binary_mask_path}'
            label_tensor[..., class_id - 1] = binary_mask[..., 0]

        # Merge all binary masks into a single-layer multiclass mask if needed
        if self.class_priority:
            label_tensor = self.aggregate_merge(label_tensor)

        return label_tensor

    def aggregate_merge(self, masks):
        final_mask = np.zeros(shape=self.image_shape, dtype='int32')
        # Start with the lowest priority class
        for class_ind in reversed(self.class_priority):
            layer = masks[..., class_ind - 1]
            untouched_area = (layer == 0).astype('int32')
            final_mask = final_mask * untouched_area + layer * class_ind
        return final_mask
