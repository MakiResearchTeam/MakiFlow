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

from makiflow.models import NeuralRender
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
    name = 'name'
    path_to_arch = 'path to arch'
    pretrained_layers = 'pretrained layers'
    weights = 'weights'
    untrainable_layers = 'untrainable layers'
    epochs = 'epochs'
    test_period = 'test period'
    save_period = 'save period'
    loss_type = 'loss type'
    l1_reg = 'l1 reg'
    l1_reg_layers = 'l1 reg layers'
    l2_reg = 'l2 reg'
    l2_reg_layers = 'l2 reg layers'
    optimizers = 'optimizers'
    iterations = 'iterations'


class SubExpField:
    opt_info = 'opt_info'


class LossType:
    AbsLoss = 'AbsLoss'
    MaskedAbsLoss = 'MaskedAbsLoss'
    MaskedMseLoss = 'MaskedMseLoss'
    MseLoss = 'MseLoss'
    PerceptualLoss = 'PerceptualLoss'


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

    # ----------------------------------------------------------------------------------------------------------------------
    # ---------------------------------SETTING UP THE EXPERIMENTS-----------------------------------------------------------

    def start_experiments(self):
        """
        Starts all the experiments.
        """
        for experiment in self._exp_params['experiments']:
            self._start_exp(experiment)

    def _create_experiment_folder(self, name):
        self._exp_folder = os.path.join(
            self._path_to_save, name
        )
        os.makedirs(self._exp_folder, exist_ok=True)

    # ----------------------------------------------------------------------------------------------------------------------
    # ---------------------------------START THE EXPERIMENTS----------------------------------------------------------------

    def _start_exp(self, experiment):
        self._create_experiment_folder(experiment[ExpField.name])
        exp_params = {
            ExpField.name: experiment[ExpField.name],
            ExpField.pretrained_layers: experiment[ExpField.pretrained_layers],
            ExpField.weights: experiment[ExpField.weights],
            ExpField.untrainable_layers: experiment[ExpField.untrainable_layers],
            ExpField.epochs: experiment[ExpField.epochs],
            ExpField.test_period: experiment[ExpField.test_period],
            ExpField.save_period: experiment[ExpField.save_period],
            ExpField.loss_type: experiment[ExpField.loss_type],
            ExpField.l1_reg: experiment[ExpField.l1_reg],
            ExpField.l1_reg_layers: experiment[ExpField.l1_reg_layers],
            ExpField.l2_reg: experiment[ExpField.l2_reg],
            ExpField.l2_reg_layers: experiment[ExpField.l2_reg_layers]
        }
        for opt_info in experiment[ExpField.optimizers]:
            exp_params[SubExpField.opt_info] = opt_info
            self._run_experiment(exp_params)

    # ----------------------------------------------------------------------------------------------------------------------
    # -----------------------------------PREPARING THE EXPERIMENT-----------------------------------------------------------

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

        model = self._model_creation_function()
        self.generator = model._generator
        self.loss_list = []

        weights_path = exp_params[ExpField.weights]
        pretrained_layers = exp_params[ExpField.pretrained_layers]
        untrainable_layers = exp_params[ExpField.untrainable_layers]

        model.set_session(self._sess)
        if weights_path is not None:
            model.load_weights(weights_path, layer_names=pretrained_layers)

        if untrainable_layers is not None:
            layers = []
            for layer_name in untrainable_layers:
                layers += [(layer_name, False)]
            model.set_layers_trainable(layers)

        # Set l1 regularization
        l1_reg = exp_params[ExpField.l1_reg]
        if l1_reg is not None:
            l1_reg = np.float32(l1_reg)
            l1_reg_layers = exp_params[ExpField.l1_reg_layers]
            reg_config = [(layer, l1_reg) for layer in l1_reg_layers]
            model.set_l1_reg(reg_config)

        # Set l2 regularization
        l2_reg = exp_params[ExpField.l2_reg]
        if l2_reg is not None:
            l2_reg = np.float32(l2_reg)
            l2_reg_layers = exp_params[ExpField.l2_reg_layers]
            reg_config = [(layer, l2_reg) for layer in l2_reg_layers]
            model.set_l2_reg(reg_config)

        return model

    # ----------------------------------------------------------------------------------------------------------------------
    # -----------------------------------EXPERIMENT UTILITIES---------------------------------------------------------------

    def _perform_testing(self, model, exp_params, epoch, FPS=25):
        # COLLECT PREDICTIONS
        print('Testing the model...')
        print('Collecting predictions...')

        pass

    # ----------------------------------------------------------------------------------------------------------------------
    # ------------------------------------EXPERIMENT LOOP-------------------------------------------------------------------

    def _run_experiment(self, exp_params):
        model = self._restore_model(exp_params)

        loss_type = exp_params[ExpField.loss_type]
        opt_info = exp_params[SubExpField.opt_info]
        epochs = exp_params[ExpField.epochs]
        iterations = exp_params[ExpField.iterations]
        test_period = exp_params[ExpField.test_period]
        save_period = exp_params[ExpField.save_period]
        optimizer, global_step = OptimizerBuilder.build_optimizer(opt_info)
        if global_step is not None:
            self._sess.run(tf.variables_initializer([global_step]))

        # Catch InterruptException
        try:
            for i in range(epochs):
                if self.generator is None:
                    if loss_type == LossType.AbsLoss:
                        sub_train_info = None
                    elif loss_type == LossType.MseLoss:
                        sub_train_info = None
                    else:
                        raise ValueError('Unknown loss type!')
                else:
                    if loss_type == LossType.AbsLoss:
                        sub_train_info = model.gen_fit_abs(optimizer=optimizer, epochs=1,
                                                           iterations=iterations, global_step=global_step)
                    elif loss_type == LossType.MseLoss:
                        sub_train_info = model.gen_fit_mse(optimizer=optimizer, epochs=1,
                                                           iterations=iterations, global_step=global_step)
                    elif loss_type == LossType.MaskedAbsLoss:
                        sub_train_info = model.gen_fit_masked_abs(optimizer=optimizer, epochs=1,
                                                                  iterations=iterations, global_step=global_step)
                    elif loss_type == LossType.MaskedMseLoss:
                        sub_train_info = model.gen_fit_masked_mse(optimizer=optimizer, epochs=1,
                                                                  iterations=iterations, global_step=global_step)
                    elif loss_type == LossType.PerceptualLoss:
                        sub_train_info = model.gen_fit_perceptual(optimizer=optimizer, epochs=1,
                                                                  iterations=iterations, global_step=global_step)
                    else:
                        raise ValueError('Unknown loss type!')

                self.loss_list += sub_train_info['train losses']

                if i % test_period == 0:
                    self._perform_testing(model, exp_params, i)

                if save_period is not None and i % save_period == 0:
                    os.makedirs(
                        f'{self._path_to_save}/epoch_{i}/', exist_ok=True
                    )
                    model.save_weights(f'{self._path_to_save}/epoch_{i}/weights.ckpt')
                print('Epochs:', i)
        except KeyboardInterrupt as ex:
            traceback.print_exc()
            print("SAVING GAINED DATA")
        finally:
            # ALWAYS DO LAST SAVE
            os.makedirs(
                f'{self._path_to_save}/last_weights/', exist_ok=True
            )
            model.save_weights(f'{self._path_to_save}/last_weights/weights.ckpt')
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

            self._create_loss_info(loss_type)
            print('Sub test is done.')

    # ----------------------------------------------------------------------------------------------------------------------
    # -----------------------------------SAVING TRAINING RESULTS------------------------------------------------------------

    def _create_loss_info(self, loss_type):
        """
        # Plot all dices
        TestVisualizer.plot_test_values(
            test_values=values[1:],
            legends=labels[1:],
            x_label='Epochs',
            y_label='Dice',
            save_path=f'{self._path_to_save}/dices.png'
        )
        """

        # Plot Loss
        TestVisualizer.plot_test_values(
            test_values=self.loss_list,
            legends=loss_type,
            x_label='Epochs',
            y_label='Loss',
            save_path=f'{self._path_to_save}/loss.png'
        )

