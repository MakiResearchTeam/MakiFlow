import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TableParser:
    """
    Class that can allocate the table and align, crop it
    """

    def __init__(self, label_tables, scan_image_w=2550, scan_image_h=3501, kernel_size=(5, 5),
                 min_val=50, max_val=150, aperture_size=3, l2gradient=False,
                 morph_feat=cv2.MORPH_CROSS,
                 retr_feat=cv2.RETR_EXTERNAL, chain_approx_feat=cv2.CHAIN_APPROX_NONE,
                 approx_const=0.1):
        """
        Here will be calculated the size of one percent of image

        Parameters
        ----------
            label_tables : array
                list with words, what contains the cropped cells
            retr_feat :
                parameter for Canny (watch cv2.findContours)
            chain_approx_feat :
                parameter for Canny (watch cv2.findContours)
            morph_feat :
                parameter for Canny (watch cv2.morphologyEx)
            l2gradient : boolean
                parameter for Canny (watch cv2.Canny)
            aperture_size : int
                parameter for Canny (watch cv2.Canny)
            max_val : int
                parameter for Canny (watch cv2.Canny)
            min_val : int
                parameter for Canny (watch cv2.Canny)
            scan_image_w : int
                width of initial image
            scan_image_h : int
                height of initial image
        """
        self.label_tables = label_tables
        self.table = None
        self.min_val = min_val
        self.max_val = max_val
        self.aperture_size = aperture_size
        self.l2gradient = l2gradient
        self.morph_feat = morph_feat
        self.retr_feat = retr_feat
        self.chain_approx_feat = chain_approx_feat
        self.approx_const = approx_const
        self.scan_image_w = scan_image_w
        self.scan_image_h = scan_image_h
        self.scan_area_percent = scan_image_w * scan_image_h * .01
        self.kernel = np.ones(kernel_size, np.uint8)
        self.counter = 1
        pass

    def __create_and_open_table(self):
        table = pd.DataFrame({'path': [], 'feature': []})
        table.columns = ['path', 'feature']
        self.table = table
        pass

    def __put_row(self, path, feature):
        self.table.loc[len(self.table)] = {'path': path, 'feature': feature}
        pass

    def show_image(self, image, color_scheme='gray'):
        """
        This method will be show image for you

        Parameters
        ----------

            color_scheme : string
                color scheme for output, default is gray
            image : array
                image that need to show
        """
        plt.imshow(image, cmap=color_scheme)
        plt.show()
        pass

    def get_contours(self, image):
        """
        This method will highlightes lines on the image

        Parameters
        ----------

            image : array
                initial image

        Returns
        -------
            array
                binary image with highlighted lines
        """
        edges = cv2.Canny(image, self.min_val, self.max_val, apertureSize=self.aperture_size,
                          L2gradient=self.l2gradient)
        morph = cv2.morphologyEx(edges, self.morph_feat, self.kernel)
        _, contours, _ = cv2.findContours(morph, self.retr_feat, self.chain_approx_feat)
        return contours

    def get_table_bounds(self, image):
        """
        Find all bounding boxes on image

        Parameters
        ----------

            image : array
                initial image

        Returns
        -------
            array
                image with all checkboxes
        """
        contours = self.get_contours(image)

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 5)
        return image

    def get_approx_table_bound(self, image, approx_const):
        """
        Find best fit table bounding box

        Parameters
        ----------

            image : array
                initial image
            approx_const : float
                const that needed to approximate bounding box better (just parameter what need to find for each task)

        Returns
        -------
            array
                bounding box
        """

        contours = self.get_contours(image)

        for contour in contours:
            area = cv2.contourArea(contour)
            epsilon = approx_const * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) > 3 and 50 < area / self.scan_area_percent < 70:
                return approx

        return self.get_approx_table_bound(image, approx_const=self.approx_const / 2)

    def draw_approx_table_bound(self, image, approx_const=None):
        """
        Draw approx bounding box on image

        Parameters
        ----------

            image : array
                initial image
            approx_const : float
                 const that needed to approximate bounding box better (just parameter what need to find for each task)

        Returns
        -------
            array
                image with painted bounding box
        """
        if approx_const is None:
            approx_const = self.approx_const
        approx = self.get_approx_table_bound(image, approx_const)
        image = cv2.polylines(image, [approx], True, (0, 0, 0), thickness=5)
        return image

    def image_alignment(self, image, approx_const=None):
        """
        Align table on image

        Parameters
        ----------

            image : array
                initial image
            approx_const : float
                 const that needed to approximate bounding box better (just parameter what need to find for each task)

        Returns
        -------
            (array, array)
                tuple with bounding box of table and image
        """
        rows, cols = image.shape

        if approx_const is None:
            approx_const = self.approx_const
        approx = self.get_approx_table_bound(image, approx_const)

        # Dummy sort =)
        approx = sorted(approx, key=lambda x: int(x[0][0]))
        temp = sorted(approx[:2], key=lambda x: int(x[0][1]))
        temp.extend(approx[2:])
        approx = temp
        temp = sorted(approx[2:], key=lambda x: int(x[0][1]), reverse=True)
        approx = approx[:2]
        approx.extend(temp)

        src = np.float32([approx[0][0], approx[1][0], approx[2][0]])
        dst = np.float32([approx[0][0], [approx[0][0][0], approx[1][0][1]], [approx[2][0][0], approx[1][0][1]]])
        approx = [approx[0][0], [approx[0][0][0], approx[1][0][1]], [approx[2][0][0], approx[1][0][1]]]

        transpose_matrix = cv2.getAffineTransform(src, dst)
        image = cv2.warpAffine(image, transpose_matrix, (cols, rows))
        return approx, image

    def get_lines(self, image):
        """
        This method will return image with highlighted lines

        Parameters
        ----------

            image : array
                initial image
        Returns
        -------
            array
                image with highlighted lines
        """
        edges = cv2.Canny(image, self.min_val, self.max_val, apertureSize=self.aperture_size,
                          L2gradient=self.l2gradient)
        edges = cv2.morphologyEx(edges, self.morph_feat, self.kernel)
        lines = cv2.HoughLinesP(edges, rho=2, theta=np.pi / 180, threshold=400, minLineLength=1000, maxLineGap=100)
        return lines

    def crop_tables(self, src, dst):
        """
        This method will crop all tables from src and save them to dst

        Parameters
        ----------
            src : string
                path to folder where hold initial images
            dst : string
                path where images would be save
        """
        for root, dirs, files in os.walk(src):
            for dir in dirs:
                current_path = os.path.join(root, dir)
                for _, _, current_file_list in os.walk(current_path):
                    for current_file in current_file_list:
                        img = cv2.imread(os.path.join(current_path, current_file), 0)
                        box, img = self.image_alignment(img)
                        table_img = img[box[0][1]:box[1][1], box[0][0]:box[2][0]]
                        cv2.imwrite(os.path.join(f'{dst}', f'{dir}', f'{current_file}'), table_img)
        pass

    def crop_cells(self, src, dst):
        """
        This method will crop all cells from src and save them to dst. Same will be created table with dict
         {file: label}

        Parameters
        ----------
            src : string
                path to folder where hold initial images
            dst : string
                path where images would be save
        """
        num_of_table = -1
        self.__create_and_open_table()

        for root, dirs, files in os.walk(src):
            for dir in dirs:
                current_path = os.path.join(root, dir)
                num_of_table += 1
                for _, _, current_file_list in os.walk(current_path):
                    for current_file in current_file_list:
                        counter = 1
                        img = cv2.imread(os.path.join(current_path, current_file), 0)

                        step_by_x = int(img.shape[1] / 4)
                        step_by_y = int(img.shape[0] / 9)
                        dy = int(step_by_y * 97 / (97 + 41))
                        title_dy = int(step_by_y * 41 / (97 + 41))

                        for i in range(9):
                            for j in range(4):
                                self.__put_row(os.path.join(f'{dst}', f'{dir}', f'{current_file[:-4]}_{counter}.png'),
                                               self.label_tables[num_of_table][i + j])
                                cropped_image = img[step_by_y * i + title_dy:step_by_y * i + title_dy + dy,
                                                step_by_x * j: step_by_x * (j + 1)]
                                cv2.imwrite(os.path.join(f'{dst}', f'{dir}', f'{current_file[:-4]}_{counter}.png'),
                                            cropped_image)
                                counter += 1
        self.table.to_csv('result', index=False)
        pass
