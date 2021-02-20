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
import glob
import os
from makiflow.gyms.gyms_modules.gyms_collector import GymCollector, SEGMENTATION, TESTER
from makiflow.gyms.gyms_modules.segmentator_gym import SegmentatorTester
from makiflow.metrics import categorical_dice_coeff
from makiflow.metrics import confusion_mat
from sklearn.metrics import f1_score
import cv2
import numpy as np


class SegmentatorBinaryTester(SegmentatorTester):
    THREASHOLD = 0.4
    # TODO: add threashold???

    def __init_train_images(self):
        if not isinstance(self._config[SegmentatorBinaryTester.TRAIN_IMAGE], list):
            train_images_path = [self._config[SegmentatorBinaryTester.TRAIN_IMAGE]]
        else:
            train_images_path = self._config[SegmentatorBinaryTester.TRAIN_IMAGE]

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
                orig_mask = self._preprocess_masks(self._train_masks_path[i])
                self._train_masks_np.append(orig_mask.astype(np.uint8))
                n_images += orig_mask.shape[-1]

            self._names_train.append(SegmentatorBinaryTester.TEST_N.format(i))
            self.add_image(self._names_train[-1], n_images=n_images)

    def __init_test_images(self):
        self._test_masks_path = self._config[self.TEST_MASK]
        if not isinstance(self._config[SegmentatorBinaryTester.TEST_IMAGE], list):
            test_images_path = [self._config[SegmentatorBinaryTester.TEST_IMAGE]]
        else:
            test_images_path = self._config[SegmentatorBinaryTester.TEST_IMAGE]

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
                orig_mask = self._preprocess_masks(self._test_masks_path[i])
                self._test_mask_np.append(orig_mask.astype(np.uint8))
                n_images += orig_mask.shape[-1]

            self._names_test.append(SegmentatorBinaryTester.TEST_N.format(i))
            # Image + orig mask (if was given) + prediction
            self.add_image(self._names_test[-1], n_images=n_images)
        if self._test_masks_path is not None:
            # Add confuse matrix image
            self._names_test += [self.CONFUSE_MATRIX]
            self.add_image(self._names_test[-1])

    def __get_train_tb_data(self, model, dict_summary_to_tb):
        if self._train_masks_path is not None:
            for i, (single_norm_train, single_train, single_mask_np) in enumerate(
                    zip(self._norm_images_train, self._train_images, self._train_masks_np)
            ):
                # If original masks were provided
                prediction = model.predict(np.stack([single_norm_train] * model.get_batch_size(), axis=0))[0]
                prediction = (prediction > SegmentatorBinaryTester.THREASHOLD).astype(np.uint8)
                array_ans = [single_train]
                for indx in range(single_mask_np.shape[-1]):
                    array_ans += [
                        self.draw_heatmap(single_mask_np[..., indx], self._names_train[i] + f'_truth_{i}'),
                        self.draw_heatmap(prediction[..., indx], self._names_train[i] + f'_{i}')
                    ]
                dict_summary_to_tb.update(
                    {
                        self._names_train[i]: np.stack(array_ans).astype(np.uint8)
                    }
                )

        else:
            for i, (single_norm_train, single_train) in enumerate(zip(self._norm_images_train, self._train_images)):
                # If there is not original masks
                # Just vis. input image and predicted mask
                prediction = model.predict(np.stack([single_norm_train] * model.get_batch_size(), axis=0))[0]
                prediction = (prediction > SegmentatorBinaryTester.THREASHOLD).astype(np.uint8)
                array_ans = [single_train]
                for indx in range(prediction.shape[-1]):
                    array_ans += [self.draw_heatmap(prediction[..., indx], self._names_train[i] + f'_truth_{i}')]
                dict_summary_to_tb.update(
                    {
                        self._names_train[i]: np.stack(array_ans).astype(np.uint8)
                    }
                )

    def __get_test_tb_data(self, model, dict_summary_to_tb, path_save_res):
        if self._test_masks_path is not None:
            all_pred = []
            for i, (single_norm_test, single_test, single_mask_np) in enumerate(
                    zip(self._test_norm_images, self._test_images, self._test_mask_np)
            ):
                # If original masks were provided
                prediction = model.predict(np.stack([single_norm_test] * model.get_batch_size(), axis=0))[0]
                prediction = (prediction > SegmentatorBinaryTester.THREASHOLD).astype(np.uint8)
                array_ans = [single_test]
                for indx in range(single_mask_np.shape[-1]):
                    array_ans += [
                        self.draw_heatmap(single_mask_np[..., indx], self._names_test[i] + f'_truth_{i}'),
                        self.draw_heatmap(prediction[..., indx], self._names_test[i] + f'_{i}')
                    ]
                dict_summary_to_tb.update(
                    {
                        self._names_test[i]: np.stack(array_ans).astype(np.uint8)
                    }
                )
            # Confuse matrix
            labels = np.array(self._test_mask_np).astype(np.uint8)
            pred_np = np.stack(all_pred[:len(labels)], axis=0).astype(np.float32)
            mat_img, res_dices_dict = self._v_dice_calc_and_confuse_m(pred_np, labels, path_save_res)
            dict_summary_to_tb.update({ self._names_test[-1]: np.expand_dims(mat_img.astype(np.uint8), axis=0) })
            dict_summary_to_tb.update(res_dices_dict)
            # f1 score
            # TODO: Delete or refactor
            labels = np.array(self._test_mask_np).astype(np.uint8)
            pred_np = np.argmax(np.stack(all_pred[:len(labels)], axis=0), axis=-1).astype(np.uint8)
            f1_score_np = f1_score(labels.reshape(-1), pred_np.reshape(-1), average='micro')
            dict_summary_to_tb.update({ SegmentatorBinaryTester.F1_SCORE: f1_score_np})
        else:
            for i, (single_norm_test, single_test) in enumerate(zip(self._test_norm_images, self._test_images)):
                # If there is not original masks
                # Just vis. input image and predicted mask
                prediction = model.predict(np.stack([single_norm_test] * model.get_batch_size(), axis=0))[0]
                prediction = (prediction > SegmentatorBinaryTester.THREASHOLD).astype(np.uint8)
                array_ans = [single_test]
                for indx in range(prediction.shape[-1]):
                    array_ans += [self.draw_heatmap(prediction[..., indx], self._names_test[i] + f'_{i}')]

                dict_summary_to_tb.update(
                    {
                        self._names_test[i]: np.stack(array_ans).astype(np.uint8)
                    }
                )

    def _preprocess_masks(self, path_mask_folder: str):
        if super()._resize_to is None:
            first_img = cv2.imread(glob.glob(os.path.join(path_mask_folder, '*.bmp'))[0])
            labels = np.zeros((
                first_img.shape[0],
                first_img.shape[1],
                len(self._config[SegmentatorTester.CLASSES_NAMES])
            )).astype(np.uint8)
        else:
            labels = np.zeros((
                *super()._resize_to,
                len(self._config[SegmentatorTester.CLASSES_NAMES])
            )).astype(np.uint8)

        for i in range(len(self._config[SegmentatorTester.CLASSES_NAMES])):
            single_label = cv2.imread(os.path.join(path_mask_folder, f'{i+1}.bmp'))
            if single_label is not None:
                _, labels[..., i] = super()._preprocess(single_label, mask_preprocess=True)

        return labels

    def _v_dice_calc_and_confuse_m(self, predictions, labels, save_folder):
        """
        Returns
        -------
        confuse_matrix: np.ndarray
        res_dices_dict : dict

        """
        print('Computing V-Dice...')
        # COMPUTE DICE AND CREATE CONFUSION MATRIX
        # TODO: Refactor method `categorical_dice_coeff`
        v_dice_val, dices = categorical_dice_coeff(predictions, labels, use_argmax=True)
        str_to_save_vdice = "V-DICE:\n"
        print('V-Dice:', v_dice_val)

        res_dices_dict = {SegmentatorBinaryTester.PREFIX_CLASSES.format(SegmentatorBinaryTester.V_DICE): v_dice_val}
        self.dices_for_each_class[SegmentatorBinaryTester.V_DICE] += [v_dice_val]

        for i, class_name in enumerate(self._config[SegmentatorBinaryTester.CLASSES_NAMES]):
            self.dices_for_each_class[class_name] += [dices[i]]
            res_dices_dict[SegmentatorBinaryTester.PREFIX_CLASSES.format(class_name)] = dices[i]
            print(f'{class_name}: {dices[i]}')
            str_to_save_vdice += f'{class_name}: {dices[i]}\n'

        with open(os.path.join(save_folder, SegmentatorBinaryTester.VDICE_TXT), 'w') as fp:
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


GymCollector.update_collector(SEGMENTATION, TESTER, SegmentatorBinaryTester)

