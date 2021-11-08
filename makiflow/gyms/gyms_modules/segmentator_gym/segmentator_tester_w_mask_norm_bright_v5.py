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

from makiflow.tools.preprocess import preprocess_input
from makiflow.metrics import categorical_dice_coeff
from makiflow.tools.test_visualizer import TestVisualizer
from makiflow.metrics.utils import one_hot
from sklearn.metrics import f1_score
import pandas as pd

from makiflow.tools.image_tools import apply_op_norm_bright
from makiflow.gyms.core import TesterBase
from makiflow.gyms.gyms_modules.gyms_collector import GymCollector, SEGMENTATION, TESTER
from makiflow.gyms.gyms_modules.segmentator_gym.utils import draw_heatmap
from makiflow.metrics import confusion_mat
import cv2
import numpy as np

W_CROP, H_CROP = 900, 900
MODEL_INPUT_SIZE = (1024, 1024)
CLASS_99_MAP_TO = 10 # 13

SIGMA_X = 30.0

# Add new preprocess
# For images


class SegmentatorTesterWMaskNormBrightV5(TesterBase):
    TEST_IMAGE = 'test_image'
    TRAIN_IMAGE = 'train_image'
    TEST_MASK = 'test_mask'
    TRAIN_MASK = 'train_mask'
    F1_SCORE = 'f1_score'
    PREFIX_CLASSES = 'V-Dice info/{}'
    CLASSES_NAMES = 'classes_names'
    V_DICE = 'V_Dice'
    VDICE_TXT = 'v_dice.txt'

    THREASH_HOLD = 'threash_hold'
    THREASHOLD_DEFAULT = 0.5
    CLASS_PRIORITY = 'class_priority'

    TRAIN_N = 'train_{}_{}'
    TEST_N = "test_{}_{}"
    CONFUSE_MATRIX = 'confuse_matrix'
    ITERATION_COUNTER = 'iteration_counter'

    _EXCEPTION_IMAGE_WAS_NOT_FOUND = "Image by path {0} was not found!"
    _CENTRAL_SIZE = 600

    def _init(self):
        self._class_priority = self._config.get(SegmentatorTesterWMaskNormBrightV5.CLASS_PRIORITY)
        assert self._class_priority is not None, "class_priority parameter has not" \
                                                 " been provided in the configuration file."
        # Add sublists for each class
        self.add_scalar(SegmentatorTesterWMaskNormBrightV5.F1_SCORE)
        self.dices_for_each_class = {SegmentatorTesterWMaskNormBrightV5.V_DICE: []}
        self.add_scalar(SegmentatorTesterWMaskNormBrightV5.PREFIX_CLASSES.format(SegmentatorTesterWMaskNormBrightV5.V_DICE))
        for class_name in self._config[SegmentatorTesterWMaskNormBrightV5.CLASSES_NAMES]:
            if int(class_name) == 99:
                continue
            else:
                class_name = str(class_name)
            self.dices_for_each_class[class_name] = []
            self.add_scalar(SegmentatorTesterWMaskNormBrightV5.PREFIX_CLASSES.format(class_name))
        # Test images
        self._init_test_images()
        # Train images
        self.add_scalar(SegmentatorTesterWMaskNormBrightV5.ITERATION_COUNTER)
        self._thr_hold = self._config.get(SegmentatorTesterWMaskNormBrightV5.THREASH_HOLD, SegmentatorTesterWMaskNormBrightV5.THREASHOLD_DEFAULT)

    def evaluate(self, model, iteration, path_save_res):
        dict_summary_to_tb = {SegmentatorTesterWMaskNormBrightV5.ITERATION_COUNTER: iteration}
        # Draw test images
        # Write heatmap,paf and image itself for each image in `_test_images`
        self._get_test_tb_data(model, dict_summary_to_tb, path_save_res)
        # Write data into tensorBoard
        self.write_summaries(
            summaries=dict_summary_to_tb,
            step=iteration
        )

    def _get_test_tb_data(self, model, dict_summary_to_tb, path_save_res):
        all_pred = []
        for i, (single_norm_train, single_train, single_mask_np) in enumerate(
                zip(self._test_norm_images, self._test_images, self._test_mask_np)
        ):
            print(f'{i+1} / {len(self._test_images)}')
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
        print('Start calculate v-dice')
        labels = np.array(self._test_mask_np).astype(np.uint8)
        print('Labels are ready with shape as: ', labels.shape)
        print('Len preds: ', len(all_pred), ' shape single element: ', all_pred[0].shape)
        pred_np = np.asarray(all_pred[:len(labels)], dtype=np.float32)
        print('Preds are ready with shape as: ', pred_np.shape)
        mat_img, res_dices_dict = self._v_dice_calc_and_confuse_m(pred_np, labels, path_save_res)
        dict_summary_to_tb.update({ self._names_test[-1]: np.expand_dims(mat_img.astype(np.uint8), axis=0) })
        dict_summary_to_tb.update(res_dices_dict)
        # f1 score
        labels = np.array(self._test_mask_np).astype(np.uint8)
        pred_np = np.argmax(np.asarray(all_pred[:len(labels)], dtype=np.float32), axis=-1).astype(np.uint8, copy=False)
        f1_score_np = f1_score(labels.reshape(-1), pred_np.reshape(-1), average='micro')
        dict_summary_to_tb.update({ SegmentatorTesterWMaskNormBrightV5.F1_SCORE: f1_score_np})

    def _preprocess(self, data, mask_preprocess=False, use_resize=True):
        if isinstance(data, str):
            image = cv2.imread(data)
            if image is None:
                raise TypeError(
                    SegmentatorTesterWMaskNormBrightV5._EXCEPTION_IMAGE_WAS_NOT_FOUND.format(data)
                )
        else:
            image = data

        if self._resize_to is not None and use_resize:
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
        image = apply_op_norm_bright(image)
        if self._norm_mode is not None and not mask_preprocess:
            image = preprocess_input(
                    image.astype(np.float32, copy=False),
                    mode=self._norm_mode
            )
        elif self._norm_div is not None or self._norm_shift is not None and not mask_preprocess:
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
        good_regions = labels != 99

        num_classes = predictions.shape[-1]
        batch_size = len(predictions)
        # Zero bad pixels
        predictions = (predictions * np.expand_dims(good_regions, axis=-1)).argmax(axis=-1)
        predictions = predictions.reshape(-1)
        predictions = one_hot(predictions, depth=num_classes)
        predictions = predictions.reshape(batch_size, -1, num_classes)
        labels = labels.reshape(batch_size, -1)

        # Remove bad pixels
        # For each image
        good_regions = labels != 99
        preds_list = []
        preds_argmax_list = []
        labels_list = []
        for i in range(batch_size):
            good_regions_s = good_regions[i]
            preds_s = predictions[i]
            label_s = labels[i]
            # Slice
            preds_s = preds_s[good_regions_s]
            label_s = label_s[good_regions_s]
            preds_list.append(preds_s)
            labels_list.append(label_s)
            preds_argmax_list.append(preds_s.argmax(axis=-1))

        v_dice_val, dices = categorical_dice_coeff(
            preds_list, labels_list, use_argmax=False,
            num_classes=num_classes, reshape=False
        )
        str_to_save_vdice = f"V-DICE: {v_dice_val}\n"
        print('V-Dice:', v_dice_val)

        res_dices_dict = {SegmentatorTesterWMaskNormBrightV5.PREFIX_CLASSES.format(SegmentatorTesterWMaskNormBrightV5.V_DICE): v_dice_val}
        self.dices_for_each_class[SegmentatorTesterWMaskNormBrightV5.V_DICE] += [v_dice_val]

        for i, class_name in enumerate(self._config[SegmentatorTesterWMaskNormBrightV5.CLASSES_NAMES]):
            if int(class_name) == 99:
                continue
            self.dices_for_each_class[class_name] += [dices[i]]
            res_dices_dict[SegmentatorTesterWMaskNormBrightV5.PREFIX_CLASSES.format(class_name)] = dices[i]
            print(f'{class_name}: {dices[i]}')
            str_to_save_vdice += f'{class_name}: {dices[i]}\n'

        with open(os.path.join(save_folder, SegmentatorTesterWMaskNormBrightV5.VDICE_TXT), 'w') as fp:
            fp.write(str_to_save_vdice)
        # Compute and save matrix
        conf_mat_path = os.path.join(save_folder,  f'mat.png')
        print('Computing confusion matrix...')
        confusion_mat(
            np.asarray(np.concatenate([elem.reshape(-1) for elem in preds_argmax_list], axis=0), dtype=np.uint8),
            np.asarray(np.concatenate([elem.reshape(-1) for elem in labels_list      ], axis=0), dtype=np.uint8),
            use_argmax_p=False, to_flatten=True,
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

    def _init_test_images(self):
        self._test_masks_path = self._config[self.TEST_MASK]
        if not isinstance(self._config[SegmentatorTesterWMaskNormBrightV5.TEST_IMAGE], list):
            test_images_path = [self._config[SegmentatorTesterWMaskNormBrightV5.TEST_IMAGE]]
        else:
            test_images_path = self._config[SegmentatorTesterWMaskNormBrightV5.TEST_IMAGE]

        self._test_norm_images = []
        self._test_images = []
        self._test_mask_np = []
        self._names_test = []

        for i, single_path in enumerate(test_images_path):
            images_list = []
            normed_images_list = []
            masks_list = []
            # Image
            norm_img, orig_img = self._preprocess(single_path, use_resize=False)
            orig_mask = self._preprocess_masks(self._test_masks_path[i])
            orig_h, orig_w = orig_img.shape[:2]
            w_steps = orig_w // W_CROP
            h_steps = orig_h // H_CROP
            crop_i = 0
            # Cut images into patches
            for w_i in range(w_steps):
                for h_i in range(h_steps):
                    # CRop image/normed image/mask
                    single_patch_norm = norm_img[
                       h_i * H_CROP: (h_i + 1) * H_CROP,
                       w_i * W_CROP: (w_i + 1) * W_CROP
                    ]
                    single_patch_image = orig_img[
                       h_i * H_CROP: (h_i + 1) * H_CROP,
                       w_i * W_CROP: (w_i + 1) * W_CROP
                    ]
                    single_patch_mask = orig_mask[
                       h_i * H_CROP: (h_i + 1) * H_CROP,
                       w_i * W_CROP: (w_i + 1) * W_CROP
                    ]
                    single_patch_norm = cv2.resize(single_patch_norm, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
                    single_patch_image = cv2.resize(single_patch_image, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
                    single_patch_mask = cv2.resize(single_patch_mask, MODEL_INPUT_SIZE, interpolation=cv2.INTER_NEAREST)

                    normed_images_list.append(single_patch_norm)
                    images_list.append(single_patch_image.astype(np.uint8))
                    masks_list.append(single_patch_mask.astype(np.uint8))
                    self._names_test.append(SegmentatorTesterWMaskNormBrightV5.TEST_N.format(i, crop_i))
                    # Image + orig mask (if was given) + prediction
                    self.add_image(self._names_test[-1], n_images=3)
                    crop_i += 1
            self._test_norm_images += normed_images_list
            self._test_images += images_list
            self._test_mask_np += masks_list

        if self._test_masks_path is not None:
            # Add confuse matrix image
            self._names_test += [self.CONFUSE_MATRIX]
            self.add_image(self._names_test[-1])

    def _preprocess_masks(self, path_mask_folder: str):
        first_img = cv2.imread(glob.glob(os.path.join(path_mask_folder, '*.bmp'))[0])
        h, w = first_img.shape[:2]

        n_classes = len(self._class_priority)

        labels = np.zeros((h, w, n_classes), dtype='int32')
        marked_zone = None

        present_classes = []
        for p_mask in glob.glob(os.path.join(path_mask_folder, '*')):
            filename = p_mask.split('/')[-1]
            class_id = int(filename.split('.')[0])
            if class_id == 99:
                marked_zone = cv2.imread(p_mask)
                _, marked_zone = self._preprocess(marked_zone, mask_preprocess=True, use_resize=False)
                continue
            present_classes.append(class_id)
            single_label = cv2.imread(p_mask)
            if single_label is not None:
                _, labels[..., class_id] = self._preprocess(single_label, mask_preprocess=True, use_resize=False)

        labels = self.__aggregate_merge(labels, (h, w), present_classes, marked_zone)
        return labels

    def __aggregate_merge(self, masks_tensor, mask_shape, present_classes, marked_zone):
        final_mask = np.zeros(shape=mask_shape, dtype='int32')
        # Start with the lowest priority class
        for class_ind in reversed(self._class_priority):
            if class_ind not in present_classes:
                continue
            indx = class_ind
            class_ind += 1
            layer = masks_tensor[..., indx]
            untouched_area = (layer == 0).astype('int32')
            final_mask = final_mask * untouched_area + layer * class_ind

        if marked_zone is not None:
            untouched_area = (marked_zone == 0).astype('int32')
            final_mask = final_mask * untouched_area + marked_zone * 99

        return final_mask


GymCollector.update_collector(SEGMENTATION, TESTER, SegmentatorTesterWMaskNormBrightV5)
