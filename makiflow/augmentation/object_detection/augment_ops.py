from __future__ import absolute_import
from makiflow.augmentation.base import AugmentOp, Augmentor
import cv2
import numpy as np


class FlipAugment(AugmentOp):
    FLIP_HORIZONTALLY = 1
    FLIP_VERTICALLY = 0

    def __init__(self, flip_type_list, keep_old_data=True):
        """
        Flips the image and the corresponding bounding boxes.
        Parameters
        ----------
        flip_type_list : list or tuple
            Add to final dataset image with entered type of flip
            Available options:
                FlipAugment.FLIP_HORIZONTALLY;
                FlipAugment.FLIP_VERTICALLY
        keep_old_data : bool
            Set to false if you don't want to include unaugmented images into the final data set.
        """
        super().__init__()
        self.flip_type_list = flip_type_list
        self.keep_old_data = keep_old_data

    def get_data(self):
        """
        Starts augmentation process.
        Returns
        -------
        two arrays
            Augmented images and masks.
        """
        img_w = self._img_shape[1]  # [img_h, img_w, channels]
        old_imgs, old_bboxes, old_classes = self._data.get_data()

        new_imgs, new_bboxes, new_classes = [], [], []
        for img, bboxes, classes in zip(old_imgs, old_bboxes, old_classes):
            for flip_type in self.flip_type_list:
                # Append images
                new_imgs.append(cv2.flip(img, flip_type))
                # Flip bboxes
                new_bboxs = np.copy(bboxes)
                new_bboxs[:, 0] = img_w - bboxes[:, 2]
                new_bboxs[:, 2] = img_w - bboxes[:, 0]
                new_bboxes.append(new_bboxs)
                # Append classes
                new_classes.append(classes)

        if self.keep_old_data:
            new_imgs += old_imgs
            new_bboxes += old_bboxes
            new_classes += old_classes

        return new_imgs, new_bboxes, new_classes


class ImageClipType:
    int = 'int'
    float = 'float'

    @staticmethod
    def clip(image, clip_type):
        if clip_type == ImageClipType.int:
            image = np.clip(image, 0, 255).astype(np.uint8)
        elif clip_type == ImageClipType.float:
            image = np.clip(image, 0.0, 1.0).astype(np.float32)
        else:
            raise RuntimeError(f'Unknown clip type: {clip_type}')
        return image


class ContrastBrightnessAugment(AugmentOp):
    def __init__(self, params, clip_type=ImageClipType.int, keep_old_data=True):
        """
        Adjusts brightness and contrast according to `params`
        Parameters
        ----------
        params : list
            List of tuples (alpha, beta). The pixel values will be changed using the following formula:
            new_pix_value = old_pix_value * alpha + beta.
        clip_type : str
            'int' (ImageClipType.int) - the images will be clipped within [0, 255] range.
            'float' (ImageClipType.float) - the images will be clipped within [0, 1] range.
        """
        super().__init__()
        self.params = params
        self.clip_type = clip_type
        self.keep_old_data = keep_old_data

    def get_data(self):
        old_imgs, old_bboxes, old_classes = self._data.get_data()
        new_imgs = []
        new_bboxes = old_bboxes
        new_classes = old_classes
        for alpha, beta in self.params:
            for image in old_imgs:
                new_image = image * alpha + beta
                new_image = ImageClipType.clip(new_image, self.clip_type)
                new_imgs += [new_image]

        if self.keep_old_data:
            new_imgs += old_imgs
            new_bboxes += old_bboxes
            new_classes += old_classes
        return new_imgs, new_bboxes, new_classes


class GaussianBlur(AugmentOp):
    def __init__(self, ksize=(3, 3), std_x=0.65, std_y=0.65, keep_old_data=True):
        """
        Blurs images using gaussian filter.
        Parameters
        ----------
        ksize : tuple
            Gaussian kernel size.
        std_x : float
            Gaussian kernel standard deviating in X direction. Higher the deviation, 'blurier' the image is.
        std_y : float
            Gaussian kernel standard deviating in X direction. Higher the deviation, 'blurier' the image is.
        """
        super().__init__()
        self.ksize = ksize
        self.std_x = std_x
        self.std_y = std_y
        self.keep_old_data = keep_old_data

    def get_data(self):
        old_imgs, old_bboxes, old_classes = self._data.get_data()
        new_imgs = []
        new_bboxes = old_bboxes
        new_classes = old_classes
        for image in old_imgs:
            new_imgs += [cv2.GaussianBlur(
                src=image,
                ksize=self.ksize,
                sigmaX=self.std_x,
                sigmaY=self.std_y
            )]

        if self.keep_old_data:
            new_imgs += old_imgs
            new_bboxes += old_bboxes
            new_classes += old_classes

        return new_imgs, new_bboxes, new_classes


class GaussianNoiseAugment(AugmentOp):
    def __init__(self, noise_tensors_num, mean=0.0, std=1.0, clip_type=ImageClipType.int, keep_old_data=True):
        """
        Adds gaussian noise to the images.
        Parameters
        ----------
        noise_tensors_num : int
            In order to prevent the network from learning to eject only one kind of noise
            multiple noise tensors will be used.
        mean : float
            Mean of the noise.
        std : float
            Standard deviation of the noise.
        clip_type : str
            'int' (ImageClipType.int) - the images will be clipped within [0, 255] range.
            'float' (ImageClipType.float) - the images will be clipped within [0, 1] range.
        """
        super().__init__()
        self.noise_tensors_num = noise_tensors_num
        self.mean = mean
        self.std = std
        self.clip_type = clip_type
        self.keep_old_data = keep_old_data

    def _prepare_noise_tensors(self):
        image_shape = self._img_shape
        self.noise = []
        for _ in range(self.noise_tensors_num):
            noise = np.random.randn(*image_shape) * self.std + self.mean
            self.noise += [noise.astype(np.float32)]

    def get_data(self):
        old_imgs, old_bboxes, old_classes = self._data.get_data()
        new_imgs = []
        new_bboxes = old_bboxes
        new_classes = old_classes
        for image in old_imgs:
            new_img = image + self.noise[np.random.randint(low=0, high=self.noise_tensors_num)]
            new_img = ImageClipType.clip(new_img, self.clip_type)
            new_imgs += [new_img]

        if self.keep_old_data:
            new_imgs += old_imgs
            new_bboxes += old_bboxes
            new_classes += old_classes

        return new_imgs, new_bboxes, new_classes

    def __call__(self, data: Augmentor):
        super().__call__(data)
        self._prepare_noise_tensors()
        return self
