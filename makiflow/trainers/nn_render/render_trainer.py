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

from __future__ import absolute_import
import json
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback
import cv2
import glob
import copy

from makiflow.models.nn_render.training_modules.abs_loss import ABS_LOSS
from makiflow.models.nn_render.training_modules.mse_loss import MSE_LOSS
from makiflow.models.nn_render.training_modules.masked_abs_loss import MASKED_ABS_LOSS
from makiflow.models.nn_render.training_modules.masked_mse_loss import MASKED_MSE_LOSS
from makiflow.models.nn_render.training_modules.perceptual_loss import PERCEPTUAL_LOSS


from makiflow.trainers.utils.optimizer_builder import OptimizerBuilder
from makiflow.tools.test_visualizer import TestVisualizer

"""
EXAMPLE OF THE TEST PARAMETERS:
experiment_params = {
    'name': name,
    'epochs': 50,
    'test period': 10,
    'save period': None or int,
    'optimizers': [
        { 
            type: ...
            params: {}
        }, 
        { 
            type: ...
            params: {}
        }
    ]
    'loss type': 'MakiLoss' or 'FocalLoss',
    'batch sizes': [10],
    'gammas': [2, 4],
    'class names': [],
    'weights': ../weights/weights.ckpt,
    'path to arch': path,
    'pretrained layers': [layer_name],
    'utrainable layers': [layer_name],
    'l1 reg': 1e-6 or None,
    'l1 reg layers': [layer_name],
    'l2 reg': 1e-6 or None,
    'l2 reg layers': [layer_name]
}"""


class ExpField:
    EXPERIMENTS = 'experiments'
    NAME = 'name'
    PRETRAINED_LAYERS = 'pretrained layers'
    WEIGHTS = 'weights'
    UNTRAINABLE_LAYERS = 'untrainable layers'
    EPOCHS = 'epochs'
    TEST_PERIOD = 'test period'
    LOSS_TYPE = 'loss type'
    L1_REG = 'l1 reg'
    L1_REG_LAYERS = 'l1 reg layers'
    L2_REG = 'l2 reg'
    L2_REG_LAYERS = 'l2 reg layers'
    OPTIMIZERS = 'optimizers'
    ITERATIONS = 'iterations'
    PATH_TEST_IMAGE = 'path_test_image'
    PATH_TEST_UV = 'path_test_uv'
    BATCH_SIZE = 'batch_size'

class SubExpField:
    OPT_INFO = 'opt_info'

class LossType:
    ABS_LOSS = 'AbsLoss'
    MASKED_ABS_LOSS = 'MaskedAbsLoss'
    MASKED_MSE_LOSS = 'MaskedMseLoss'
    MSE_LOSS = 'MseLoss'
    PERCEPTUAL_LOSS = 'PerceptualLoss'


class RenderTrainer:
    def __init__(self, model_creation_function, exp_params, path_to_save: str):
        self._exp_params = exp_params

        self._model_creation_function = model_creation_function

        if isinstance(exp_params, str):
            self._exp_params = self._load_exp_params(exp_params)

        self._path_to_save = path_to_save
        self._sess = None
        self.generator = None

    def _load_exp_params(self, json_path):
        with open(json_path) as json_file:
            json_value = json_file.read()
            exp_params = json.loads(json_value)
        return exp_params

    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------SETTING UP THE EXPERIMENTS------------------------------------------------------

    def start_experiments(self):
        """
        Starts all the experiments.
        """
        for experiment in self._exp_params[ExpField.EXPERIMENTS]:
            self._start_exp(experiment)

    def _create_experiment_folder(self, name):
        self._exp_folder = os.path.join(
            self._path_to_save, name
        )
        os.makedirs(self._exp_folder, exist_ok=True)

    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------START THE EXPERIMENTS-----------------------------------------------------------

    def _start_exp(self, experiment):
        self._create_experiment_folder(experiment[ExpField.NAME])
        exp_params = {
            ExpField.NAME: experiment[ExpField.NAME],
            ExpField.PRETRAINED_LAYERS: experiment[ExpField.PRETRAINED_LAYERS],
            ExpField.WEIGHTS: experiment[ExpField.WEIGHTS],
            ExpField.UNTRAINABLE_LAYERS: experiment[ExpField.UNTRAINABLE_LAYERS],
            ExpField.EPOCHS: experiment[ExpField.EPOCHS],
            ExpField.TEST_PERIOD: experiment[ExpField.TEST_PERIOD],
            ExpField.LOSS_TYPE: experiment[ExpField.LOSS_TYPE],
            ExpField.L1_REG: experiment[ExpField.L1_REG],
            ExpField.L1_REG_LAYERS: experiment[ExpField.L1_REG_LAYERS],
            ExpField.L2_REG: experiment[ExpField.L2_REG],
            ExpField.L2_REG_LAYERS: experiment[ExpField.L2_REG_LAYERS],
            ExpField.PATH_TEST_IMAGE: experiment[ExpField.PATH_TEST_IMAGE],
            ExpField.PATH_TEST_UV: experiment[ExpField.PATH_TEST_UV],
            ExpField.BATCH_SIZE: experiment[ExpField.BATCH_SIZE],
            ExpField.ITERATIONS: experiment[ExpField.ITERATIONS],
        }
        for i, opt_info in enumerate(experiment[ExpField.OPTIMIZERS]):
            exp_params[SubExpField.OPT_INFO] = opt_info
            self._run_experiment(exp_params, i)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------PREPARING THE EXPERIMENT------------------------------------------------------

    def _update_session(self):
        # Create new session and reset the default graph.
        # It is needed to free the GPU memory from the old weights.
        print('Updating the session...')
        if self._sess is not None:
            self._sess.close()
            tf.reset_default_graph()
        self._sess = tf.Session()
        print('Session updated.')

    def _restore_model(self, exp_params):
        # Create model instance and load weights if it's needed
        print('Restoring the model...')
        # Update session before creating the model because
        # _update_session also resets the default TF graph
        # what causes deletion of every computational graph was ever built.
        self._update_session()

        # model for train
        self._model = self._model_creation_function(use_gen=True, batch_size=exp_params[ExpField.BATCH_SIZE])
        self.generator = self._model._generator
        self.loss_list = []

        weights_path = exp_params[ExpField.WEIGHTS]
        pretrained_layers = exp_params[ExpField.PRETRAINED_LAYERS]
        untrainable_layers = exp_params[ExpField.UNTRAINABLE_LAYERS]

        self._model.set_session(self._sess)
        if weights_path is not None:
            self._model.load_weights(weights_path, layer_names=pretrained_layers)

        # model for test
        self._test_model = self._model_creation_function(use_gen=False, batch_size=exp_params[ExpField.BATCH_SIZE])

        weights_path = exp_params[ExpField.WEIGHTS]
        pretrained_layers = exp_params[ExpField.PRETRAINED_LAYERS]
        untrainable_layers = exp_params[ExpField.UNTRAINABLE_LAYERS]

        self._test_model.set_session(self._sess)
        if weights_path is not None:
            self._test_model.load_weights(weights_path, layer_names=pretrained_layers)


        if untrainable_layers is not None:
            layers = []
            for layer_name in untrainable_layers:
                layers += [(layer_name, False)]
            self._model.set_layers_trainable(layers)

        # Set l1 regularization
        l1_reg = exp_params[ExpField.L1_REG]
        if l1_reg is not None:
            l1_reg = np.float32(l1_reg)
            l1_reg_layers = exp_params[ExpField.L1_REG_LAYERS]
            reg_config = [(layer, l1_reg) for layer in l1_reg_layers]
            self._model.set_l1_reg(reg_config)

        # Set l2 regularization
        l2_reg = exp_params[ExpField.L2_REG]
        if l2_reg is not None:
            l2_reg = np.float32(l2_reg)
            l2_reg_layers = exp_params[ExpField.L2_REG_LAYERS]
            reg_config = [(layer, l2_reg) for layer in l2_reg_layers]
            self._model.set_l2_reg(reg_config)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------EXPERIMENT UTILITIES----------------------------------------------------------

    def _perform_testing(self, exp_params, save_path, path_to_weights, FPS=25):
        # Create test video from pure model.
        # NOTICE output of model and input image size (not UV) must be equal
        print('Testing the model...')
        print('Collecting predictions...')

        # Collect data and predictions

        uv = []
        masks = []
        origin_image = []

        for m in range(len(exp_params[ExpField.PATH_TEST_UV])):
            path = exp_params[ExpField.PATH_TEST_UV][m]
            path_image = exp_params[ExpField.PATH_TEST_IMAGE][m]
            count = len(glob.glob(path + '/*.npy'))
            for i in range(1, count):
                load = np.load(path + '/' + str(i) + '.npy')
                image = cv2.imread(path_image + '/' + str(i) + '.png')
                mask = copy.deepcopy(load[:, :, 0])
                mask[mask > 0.0] = 1.0
                masks.append(mask.astype(np.float32))
                uv.append(load.astype(np.float32))
                origin_image.append(image.astype(np.float32))

        without_face = []
        for i in range(len(origin_image)):
            mask = masks[i]
            image = copy.deepcopy(origin_image[i])
            image[:, :, 0] *= (mask != 1)
            image[:, :, 1] *= (mask != 1)
            image[:, :, 2] *= (mask != 1)
            without_face.append(image.astype(np.float32))

        batch_size = exp_params[ExpField.BATCH_SIZE]

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(f'{save_path}render_video.mp4', fourcc, FPS,
                                 (origin_image[0].shape[0] * 2, origin_image[0].shape[0]), True)

        all_pred = []

        for i in range(len(uv) // batch_size):
            answer = self._test_model.predict(uv[i * batch_size:batch_size * (i + 1)])
            all_pred += [i for i in answer]

        print('render video')
        for i in range(len(all_pred)):
            answer = np.clip(all_pred[i] + 1, 0.0, 2.0)
            answer = answer * 128
            answer[:, :, 0] = without_face[i][:, :, 0] + masks[i] * answer[:, :, 0]
            answer[:, :, 1] = without_face[i][:, :, 1] + masks[i] * answer[:, :, 1]
            answer[:, :, 2] = without_face[i][:, :, 2] + masks[i] * answer[:, :, 2]
            answer = np.concatenate([answer, origin_image[i]], axis=1)
            answer = answer.astype(np.uint8)
            writer.write(answer)

        writer.release()
        print(f'{save_path}/render_video.mp4 video was created!')
        test_ses.close()

    # -----------------------------------------------------------------------------------------------------------------
    # ------------------------------------EXPERIMENT LOOP--------------------------------------------------------------

    def _run_experiment(self, exp_params, number_of_experiment):
        self._restore_model(exp_params)

        loss_type = exp_params[ExpField.LOSS_TYPE]
        opt_info = exp_params[SubExpField.OPT_INFO]
        epochs = exp_params[ExpField.EPOCHS]
        iterations = exp_params[ExpField.ITERATIONS]
        test_period = exp_params[ExpField.TEST_PERIOD]
        optimizer, global_step = OptimizerBuilder.build_optimizer(opt_info)
        if global_step is not None:
            self._sess.run(tf.variables_initializer([global_step]))

        # Catch InterruptException
        try:
            for i in range(epochs):
                if loss_type == LossType.ABS_LOSS:
                    sub_train_info = self._model.gen_fit_abs(optimizer=optimizer, epochs=1,
                                                       iterations=iterations, global_step=global_step)
                    loss_value = sub_train_info[ABS_LOSS][0]
                elif loss_type == LossType.MSE_LOSS:
                    sub_train_info = self._model.gen_fit_mse(optimizer=optimizer, epochs=1,
                                                       iterations=iterations, global_step=global_step)
                    loss_value = sub_train_info[MSE_LOSS][0]
                elif loss_type == LossType.MASKED_ABS_LOSS:
                    sub_train_info = self._model.gen_fit_masked_abs(optimizer=optimizer, epochs=1,
                                                              iterations=iterations, global_step=global_step)
                    loss_value = sub_train_info[MASKED_ABS_LOSS][0]
                elif loss_type == LossType.MASKED_MSE_LOSS:
                    sub_train_info = self._model.gen_fit_masked_mse(optimizer=optimizer, epochs=1,
                                                              iterations=iterations, global_step=global_step)
                    loss_value = sub_train_info[MASKED_MSE_LOSS][0]
                elif loss_type == LossType.PERCEPTUAL_LOSS:
                    sub_train_info = self._model.gen_fit_perceptual(optimizer=optimizer, epochs=1,
                                                              iterations=iterations, global_step=global_step)
                    loss_value = sub_train_info[PERCEPTUAL_LOSS][0]
                else:
                    raise ValueError(f'Unknown loss type {loss_type}!')

                self.loss_list += [loss_value]

                # For generators we should save weights and then load them into new model to perform test
                if i % test_period == 0:
                    save_path = f'{self._exp_folder}/{number_of_experiment}_exp/epoch_{i}/'
                    os.makedirs(
                        save_path, exist_ok=True
                    )
                    self._model.save_weights(save_path + 'weights.ckpt')

                    self._perform_testing(exp_params, save_path, save_path + 'weights.ckpt')
                print('Epochs:', i)
        except KeyboardInterrupt as ex:
            traceback.print_exc()
            print("SAVING GAINED DATA")
        finally:
            # ALWAYS DO LAST SAVE
            save_path = f'{self._exp_folder}/{number_of_experiment}_exp'
            os.makedirs(
                save_path + '/last_weights', exist_ok=True
            )
            self._model.save_weights(f'{save_path}/last_weights/weights.ckpt')
            self._perform_testing(exp_params, save_path + '/last_weights/', save_path + '/last_weights/weights.ckpt')
            print('Test finished.')

            # Close the session since Generator yields unexpected behaviour otherwise.
            # Process doesn't stop until KeyboardInterruptExceptions occurs.
            # It also yield the following warning message:
            # 'Error occurred when finalizing GeneratorDataset iterator:
            # Failed precondition: Python interpreter state is not initialized. The process may be terminated.'
            self._sess.close()

            # Set the variable to None to avoid exceptions while closing the session again
            # in the _update_session() method.
            self._sess = None
            print('Session is closed.')

            self._create_loss_info(loss_type, save_path)
            print('Sub test is done.')

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------SAVING TRAINING RESULTS-------------------------------------------------------

    def _create_loss_info(self, loss_type, save_path):
        # Plot Loss
        TestVisualizer.plot_test_values(
            test_values=[self.loss_list],
            legends=[loss_type],
            x_label='Epochs',
            y_label='Loss',
            save_path=f'{save_path}/loss_info.png'
        )


