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

from makiflow.gyms.gyms_modules.gyms_collector import GymCollector, SEGMENTATION, TESTER
from makiflow.gyms.core import TesterBase
from makiflow.tools.preprocess import preprocess_input
from makiflow.metrics import categorical_dice_coeff
from makiflow.metrics import confusion_mat
from makiflow.tools.test_visualizer import TestVisualizer
from makiflow.gyms.gyms_modules.segmentator_gym.utils import draw_heatmap
from sklearn.metrics import f1_score
import pandas as pd
import cv2
import numpy as np



class SegmentatorTester(TesterBase):
    TEST_IMAGE = 'test_image'
    TRAIN_IMAGE = 'train_image'
    TEST_MASK = 'test_mask'
    TRAIN_MASK = 'train_mask'
    F1_SCORE = 'f1_score'
    PREFIX_CLASSES = 'V-Dice info/{}'
    CLASSES_NAMES = 'classes_names'
    V_DICE = 'V_Dice'
    VDICE_TXT = 'v_dice.txt'

    TRAIN_N = 'train_{}'
    TEST_N = "test_{}"
    CONFUSE_MATRIX = 'confuse_matrix'
    ITERATION_COUNTER = 'iteration_counter'

    _EXCEPTION_IMAGE_WAS_NOT_FOUND = "Image by path {0} was not found!"
    _CENTRAL_SIZE = 600

    def _init(self):
        # Add sublists for each class
        self.add_scalar(SegmentatorTester.F1_SCORE)
        self.dices_for_each_class = {SegmentatorTester.V_DICE: []}
        self.add_scalar(SegmentatorTester.PREFIX_CLASSES.format(SegmentatorTester.V_DICE))
        for class_name in self._config[SegmentatorTester.CLASSES_NAMES]:
            self.dices_for_each_class[class_name] = []
            self.add_scalar(SegmentatorTester.PREFIX_CLASSES.format(class_name))
        # Test images
        self._init_test_images()
        # Train images
        self._init_train_images()
        self.add_scalar(SegmentatorTester.ITERATION_COUNTER)

    def _init_train_images(self):
        if not isinstance(self._config[SegmentatorTester.TRAIN_IMAGE], list):
            train_images_path = [self._config[SegmentatorTester.TRAIN_IMAGE]]
        else:
            train_images_path = self._config[SegmentatorTester.TRAIN_IMAGE]

        self._train_masks_path = self._config[self.TRAIN_MASK]
        self._norm_images_train = []
        self._train_images = []
        self._train_masks_np = []
        self._names_train = []

        for i in range(len(train_images_path)):
            # Image
            norm_img, orig_img = self._preprocess(train_images_path[i])
            self._norm_images_train.append(
                norm_img
            )
            self._train_images.append(orig_img.astype(np.uint8))
            n_images = 2
            # Mask
            if self._train_masks_path is not None and len(self._train_masks_path) > i:
                _, orig_mask = self._preprocess(self._train_masks_path[i], mask_preprocess=True)
                self._train_masks_np.append(orig_mask.astype(np.uint8))
                n_images += 1

            self._names_train.append(SegmentatorTester.TRAIN_N.format(i))
            self.add_image(self._names_train[-1], n_images=n_images)

    def _init_test_images(self):
        self._test_masks_path = self._config[self.TEST_MASK]
        if not isinstance(self._config[SegmentatorTester.TEST_IMAGE], list):
            test_images_path = [self._config[SegmentatorTester.TEST_IMAGE]]
        else:
            test_images_path = self._config[SegmentatorTester.TEST_IMAGE]

        self._test_norm_images = []
        self._test_images = []
        self._test_mask_np = []
        self._names_test = []

        for i, single_path in enumerate(test_images_path):
            # Image
            norm_img, orig_img = self._preprocess(single_path)
            self._test_norm_images.append(
                norm_img
            )
            self._test_images.append(orig_img.astype(np.uint8))
            n_images = 2
            # Mask
            if self._test_masks_path is not None and len(self._test_masks_path) > i:
                _, orig_mask = self._preprocess(self._test_masks_path[i], mask_preprocess=True)
                self._test_mask_np.append(orig_mask.astype(np.uint8))
                n_images += 1

            self._names_test.append(SegmentatorTester.TEST_N.format(i))
            # Image + orig mask (if was given) + prediction
            self.add_image(self._names_test[-1], n_images=n_images)
        if self._test_masks_path is not None:
            # Add confuse matrix image
            self._names_test += [self.CONFUSE_MATRIX]
            self.add_image(self._names_test[-1])

    def evaluate(self, model, iteration, path_save_res):
        dict_summary_to_tb = {SegmentatorTester.ITERATION_COUNTER: iteration}
        # Draw test images
        # Write heatmap,paf and image itself for each image in `_test_images`
        self._get_test_tb_data(model, dict_summary_to_tb, path_save_res)
        # Draw train images
        self._get_train_tb_data(model, dict_summary_to_tb)
        # Write data into tensorBoard
        self.write_summaries(
            summaries=dict_summary_to_tb,
            step=iteration
        )

    def _get_train_tb_data(self, model, dict_summary_to_tb):
        if self._train_masks_path is not None:
            for i, (single_norm_train, single_train, single_mask_np) in enumerate(
                    zip(self._norm_images_train, self._train_images, self._train_masks_np)
            ):
                # If original masks were provided
                prediction = np.argmax(
                    model.predict(np.stack([single_norm_train] * model.get_batch_size(), axis=0))[0],
                    axis=-1
                )
                dict_summary_to_tb.update(
                    {
                        self._names_train[i]: np.stack(
                            [
                                single_train,
                                draw_heatmap(single_mask_np, self._names_train[i] + '_truth'),
                                draw_heatmap(prediction, self._names_train[i])
                            ]
                        ).astype(np.uint8)
                    }
                )

        else:
            for i, (single_norm_train, single_train) in enumerate(zip(self._norm_images_train, self._train_images)):
                # If there is not original masks
                # Just vis. input image and predicted mask
                prediction = np.argmax(
                    model.predict(np.stack([single_norm_train] * model.get_batch_size(), axis=0))[0],
                    axis=-1
                )
                dict_summary_to_tb.update(
                    {
                        self._names_train[i]: np.stack(
                            [single_train, draw_heatmap(prediction, self._names_train[i])]
                        ).astype(np.uint8)
                    }
                )

    def _get_test_tb_data(self, model, dict_summary_to_tb, path_save_res):
        if self._test_masks_path is not None:
            all_pred = []
            for i, (single_norm_train, single_train, single_mask_np) in enumerate(
                    zip(self._test_norm_images, self._test_images, self._test_mask_np)
            ):
                # If original masks were provided
                prediction = model.predict(np.stack([single_norm_train] * model.get_batch_size(), axis=0))[0]
                all_pred.append(prediction)
                # [..., num_classes]
                prediction_argmax = np.argmax(prediction, axis=-1)
                dict_summary_to_tb.update(
                    {
                        self._names_test[i]: np.stack(
                            [
                                single_train,
                                draw_heatmap(single_mask_np, self._names_test[i] + '_truth'),
                                draw_heatmap(prediction_argmax, self._names_test[i])
                            ]
                        ).astype(np.uint8)
                    }
                )
            # Confuse matrix
            labels = np.array(self._test_mask_np).astype(np.uint8)
            pred_np = np.stack(all_pred[:len(labels)], axis=0).astype(np.float32)
            mat_img, res_dices_dict = self._v_dice_calc_and_confuse_m(pred_np, labels, path_save_res)
            dict_summary_to_tb.update({ self._names_test[-1]: np.expand_dims(mat_img.astype(np.uint8), axis=0) })
            dict_summary_to_tb.update(res_dices_dict)
            # f1 score
            labels = np.array(self._test_mask_np).astype(np.uint8)
            pred_np = np.argmax(np.stack(all_pred[:len(labels)], axis=0), axis=-1).astype(np.uint8)
            f1_score_np = f1_score(labels.reshape(-1), pred_np.reshape(-1), average='micro')
            dict_summary_to_tb.update({ SegmentatorTester.F1_SCORE: f1_score_np})
        else:
            for i, (single_norm_train, single_train) in enumerate(zip(self._test_norm_images, self._test_images)):
                # If there is not original masks
                # Just vis. input image and predicted mask
                prediction = np.argmax(
                    model.predict(np.stack([single_norm_train] * model.get_batch_size(), axis=0))[0],
                    axis=-1
                )
                dict_summary_to_tb.update(
                    {
                        self._names_test[i]: np.stack(
                            [single_train, draw_heatmap(prediction, self._names_test[i])]
                        ).astype(np.uint8)
                    }
                )

    def _preprocess(self, data, mask_preprocess=False):
        if isinstance(data, str):
            image = cv2.imread(data)
            if image is None:
                raise TypeError(
                    SegmentatorTester._EXCEPTION_IMAGE_WAS_NOT_FOUND.format(data)
                )
        else:
            image = data

        if self._resize_to is not None:
            new_h, new_w = self._resize_to
            if mask_preprocess:
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if mask_preprocess:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif self._use_bgr2rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_img = image.copy()

        if self._norm_mode is not None:
            image = preprocess_input(
                    image.astype(np.float32, copy=False),
                    mode=self._norm_mode
            )
        elif self._norm_div is not None or self._norm_shift is not None:
            image = image.astype(np.float32, copy=False)
            if self._norm_div is not None:
                image /= self._norm_div

            if self._norm_shift is not None:
                image -= self._norm_shift

        return image.astype(np.float32, copy=False), orig_img

    def _v_dice_calc_and_confuse_m(self, predictions, labels, save_folder):
        """
        Returns
        -------
        confuse_matrix: np.ndarray
        res_dices_dict : dict

        """
        print('Computing V-Dice...')
        # COMPUTE DICE AND CREATE CONFUSION MATRIX
        v_dice_val, dices = categorical_dice_coeff(predictions, labels, use_argmax=True)
        str_to_save_vdice = "V-DICE:\n"
        print('V-Dice:', v_dice_val)

        res_dices_dict = {SegmentatorTester.PREFIX_CLASSES.format(SegmentatorTester.V_DICE): v_dice_val}
        self.dices_for_each_class[SegmentatorTester.V_DICE] += [v_dice_val]

        for i, class_name in enumerate(self._config[SegmentatorTester.CLASSES_NAMES]):
            self.dices_for_each_class[class_name] += [dices[i]]
            res_dices_dict[SegmentatorTester.PREFIX_CLASSES.format(class_name)] = dices[i]
            print(f'{class_name}: {dices[i]}')
            str_to_save_vdice += f'{class_name}: {dices[i]}\n'

        with open(os.path.join(save_folder, SegmentatorTester.VDICE_TXT), 'w') as fp:
            fp.write(str_to_save_vdice)
        # Compute and save matrix
        conf_mat_path = os.path.join(save_folder,  f'mat.png')
        print('Computing confusion matrix...')
        confusion_mat(
            predictions, labels, use_argmax_p=True, to_flatten=True,
            save_path=conf_mat_path, dpi=175
        )
        # Read img and convert it to rgb
        return cv2.imread(conf_mat_path)[..., ::-1], res_dices_dict

    def final_eval(self, path_to_save):
        test_df = pd.DataFrame(self.dices_for_each_class)
        test_df.to_csv(f'{path_to_save}/test_info.csv')
        labels = [key for key in self.dices_for_each_class]
        values = [self.dices_for_each_class[key] for key in self.dices_for_each_class]
        # Plot all dices
        TestVisualizer.plot_test_values(
            test_values=values,
            legends=labels,
            x_label='Epochs',
            y_label='Dice',
            save_path=f'{path_to_save}/dices.png'
        )


GymCollector.update_collector(SEGMENTATION, TESTER, SegmentatorTester)

