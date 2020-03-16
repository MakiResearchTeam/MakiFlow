# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

import os
from threading import Thread

from PIL import Image


class ImageCropperExecutor:

    def crop_images(self, images, dst, sliding_window, step=None, resize_img=None, resize_crop=None):
        """
        This method will create threads pool that will create new images by cropping the input image
        :param dst: output path
        :param images: list of images that need to crop
        :param sliding_window: tuple with two values: first - width of output image, second - height
        :param step: tuple with two values: first - shift sliding_window left on x px for each iteration, second -
        shift sliding_window down on x px for each iteration. If first or second value = zero, then step
        = sliding_window
        :param resize_img: if not None, initial image will be scaled by this param
        :param resize_crop: if not None, resulted images will be scaled by this param
        """
        threads_count = os.cpu_count()
        images_count = len(images)
        if threads_count > images_count:
            threads_count = 1

        if not os.path.exists(dst):
            os.mkdir(dst)

        images_count_in_thread = int(images_count / threads_count)
        for i in range(threads_count - 2):
            images_part = images[images_count_in_thread * i: images_count_in_thread * (i + 1)]
            cropper = ImageCropper(images_part, dst, sliding_window, step, resize_img, resize_crop)
            cropper.start()

        images_part = images[images_count_in_thread * (threads_count - 1): images_count]
        cropper = ImageCropper(images_part, dst, sliding_window, step, resize_img, resize_crop)
        cropper.start()
        pass

    pass


class ImageCropper(Thread):
    """
    This thread will crop your images
    """

    def __init__(self, images, dst, sliding_window, step, resize_img, resize_crop):
        super().__init__()
        self.dst = dst
        self.resize_crop = resize_crop
        self.resize_img = resize_img
        self.step = step
        if step is None:
            self.step = (sliding_window[0], sliding_window[1])
        self.sliding_window = sliding_window
        self.images = images

    def run(self):
        self.crop_all_files()
        pass

    def multiple_crop(self, image):
        image = Image.open(image)
        if self.resize_img is not None:
            temp = image.resize(self.resize_img)
            temp.filename = image.filename
            temp.format = image.format
            image = temp

        width, height = image.size
        dx = self.step[0]
        dy = self.step[1]
        w_window = self.sliding_window[0]
        h_window = self.sliding_window[1]
        temp = image.copy()

        x_repeats_count = width // dx
        y_repeats_count = height // dy

        for i in range(x_repeats_count):
            if i * dx + w_window > width:
                break
            for j in range(y_repeats_count):
                if j * dy + h_window > height:
                    break
                x1 = i * dx
                y1 = j * dy
                x2 = i * dx + w_window
                y2 = j * dy + h_window
                crop = temp.crop((x1, y1, x2, y2))
                if self.resize_crop is not None:
                    crop = crop.resize(self.resize_crop)
                crop.save(os.path.join(self.dst, f"{image.filename[:-4]}_{i}_{j}.{image.format}"))
                temp = image.copy()
        pass

    def crop_all_files(self):
        for img in self.images:
            self.multiple_crop(img)
        pass

    pass
