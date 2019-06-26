class ImageCutter:

    @staticmethod
    def get_ssd_input(image, input_size, x_step, y_step):
        """
        Cuts given image into pieces of size of the given `input_size`.
        Parameters
        ----------
        image : numpy ndarray
            Image to cut. Have the following shape: (image height, image width, color channels), i.e. cv2 format.
        input_size : tuple in two ints
            (input_width, input_height)
        x_step : int
            The method uses sliding window in order get pieces. This is 
            the size of the step on the x axis(width axis).
        y_step : int
            The method uses sliding window in order get pieces. This is 
            the size of the step on the y axis(height axis).
        Returns
        -------
        tuple of two lists
            The first list contains the image pieces, the second one contains offsets for later
            coordinates correction of the predicted bounding boxes by the SSD.
        """
        image_pieces = []
        offsets = []
        
        image = image.transpose([1, 0, 2]) # width / height
        image_shape = image.shape 
        hor_steps = (image_shape[0] - input_size[0]) // x_step
        vert_steps = (image_shape[1] - input_size[1]) // y_step
        for i in range(vert_steps):
            y_offset = i*y_step
            x_offset = 0
            for j in range(hor_steps):
                piece = image[j*x_step:j*x_step+input_size[0], i*y_step: i*y_step + input_size[1]]
                x_offset = j*x_step
                image_pieces.append(piece)
                offsets.append((x_offset, y_offset))
        return (image_pieces, offsets)




    @staticmethod
    def get_bounded_texts(image, bboxes):
        """
        Returns pieces of images bounded by `bboxes`.
        Parameters
        ----------
        image : numpy ndarray
            Image to get bounded pieces from. It has the following shape: (image height, image width).
        bboxes : list 
            Contains lists of 4 ints for two coordinates of the bouding box: left upper corner, right down corner.
            [x1, y1, x2, y2].
        Returns
        -------
        list 
            Contains cutted pieces of the image.
        """
        pieces = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            piece = image[int(y1): int(y2):, int(x1): int(x2)] # height, width
            pieces.append(piece)
        return pieces