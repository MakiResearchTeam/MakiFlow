import cv2


class ImageCutter:

    @staticmethod
    def image_and_mask_cutter(images, masks, window_h, window_w, step_x, step_y, scale_factor, postprocessing=None,
                              use_all_px=True):
        """
        Crops `images` and `masks` using sliding window with resize.
        Parameters
        ----------
        images : list
            List of input images.
        masks : list
            List of input masks.
        window_h : int
            Output image height.
        window_w : int
            Output image width.
        step_x : int
            Sliding window step by OX.
        step_y : int
            Sliding window step by OX.
        scale_factor : float
            Scale factor, must be in range (0, 1). After each 'sliding window step' the original images
            are resized to (previous_width * scale_factor, previous_height * scale_factor).
        postprocessing : func
            Post processing function, using on cropped image (may be function what calculate num positives pixels).
        use_all_px : bool
            If True, all pixels of image would be in output lists.

        Returns
        -------
        Three list:
            1. cropped images
            2. cropped masks
            3. additional list (result of post processing)
        """
        assert (0 < scale_factor < 1)
        assert (len(images) > 0)
        assert (len(images) == len(masks))
        assert (window_h > 0 and window_w > 0 and step_x > 0 and step_y > 0)

        cropped_images = []
        cropped_masks = []
        additional_list = []
        dx = 0
        dy = 0

        for index, (img, mask) in enumerate(zip(images, masks)):
            assert (img.shape == mask.shape)
            current_height, current_width = img.shape[:2]

            while current_height > window_h and current_width > window_w:

                for dy in range(int((current_height - window_h) / step_y)):
                    for dx in range(int((current_width - window_w) / step_x)):
                        crop_img, crop_mask = ImageCutter.crop_img_and_mask(
                            img,
                            mask,
                            dy * step_y, dy * step_y + window_h, dx * step_x, dx * step_x + window_w)
                        cropped_images.append(crop_img)
                        cropped_masks.append(crop_mask)
                        if postprocessing is not None:
                            additional_list.append(postprocessing(crop_img, crop_mask))

                if use_all_px:
                    overlap_y = dy * step_y + window_h != current_height
                    overlap_x = dx * step_x + window_w != current_width
                    if overlap_y:
                        for dx in range(int((current_width - window_w) / step_x)):
                            crop_img, crop_mask = ImageCutter.crop_img_and_mask(
                                img,
                                mask,
                                current_height - window_h, current_height, dx * step_x, dx * step_x + window_w)
                            cropped_images.append(crop_img)
                            cropped_masks.append(crop_mask)

                            if postprocessing is not None:
                                additional_list.append(postprocessing(crop_img, crop_mask))
                    if overlap_x:
                        for dy in range(int((current_height - window_h) / step_y)):
                            crop_img, crop_mask = ImageCutter.crop_img_and_mask(
                                img,
                                mask,
                                dy * step_y, dy * step_y + window_h, current_width - window_w, current_width)
                            cropped_images.append(crop_img)
                            cropped_masks.append(crop_mask)

                            if postprocessing is not None:
                                additional_list.append(postprocessing(crop_img, crop_mask))
                    if overlap_x and overlap_y:
                        crop_img, crop_mask = ImageCutter.crop_img_and_mask(
                            img,
                            mask,
                            current_height - window_h, current_height, current_width - window_w, current_width)
                        cropped_images.append(crop_img)
                        cropped_masks.append(crop_mask)

                        if postprocessing is not None:
                            additional_list.append(postprocessing(crop_img, crop_mask))

                img = cv2.resize(img, (int(current_width * scale_factor), int(current_height * scale_factor)))
                mask = cv2.resize(mask, (int(current_width * scale_factor), int(current_height * scale_factor)))

                current_height, current_width = img.shape[:2]

        return cropped_images, cropped_masks, additional_list

    @staticmethod
    def crop_img_and_mask(img, mask, up, down, left, right):
        crop_img = img[up: down, left: right]
        crop_mask = mask[up: down, left: right]
        return crop_img, crop_mask
