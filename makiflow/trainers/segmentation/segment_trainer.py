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
from makiflow.metrics import categorical_dice_coeff
from makiflow.trainers.utils.optimizer_builder import OptimizerBuilder
from sklearn.utils import shuffle
from makiflow.save_recover.builder import Builder
from makiflow.tools.test_visualizer import TestVisualizer
from tqdm import tqdm
import traceback

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
    class_names = 'class names'
    save_period = 'save period'
    loss_type = 'loss type'
    l1_reg = 'l1 reg'
    l1_reg_layers = 'l1 reg layers'
    l2_reg = 'l2 reg'
    l2_reg_layers = 'l2 reg layers'
    optimizers = 'optimizers'
    batch_sizes = 'batch sizes'
    gammas = 'gammas'


class SubExpField:
    opt_info = 'opt_info'
    batch_sz = 'batch_sz'
    gamma = 'gamma'


class LossType:
    FocalLoss = 'FocalLoss'
    QuadraticCELoss = 'QuadraticCELoss'
    MakiLoss = 'MakiLoss'


# SEGMENTATOR IMPLIES THAT ALL NETWORK ARCHITECTURES HAVE THE SAME INPUT SHAPE
class SegmentatorTrainer:
    def __init__(self, exp_params, path_to_save: str):
        self._exp_params = exp_params
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

    # noinspection PyAttributeOutsideInit
    def set_train_data(self, Xtrain, Ytrain, num_pos):
        """
        Parameters
        ----------
        Xtrain : list
            Training images.
        Ytrain : list
            Training masks.
        num_pos : list
            List contains number of positive sample for the corresponding images.
        """
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.num_pos = num_pos

    def set_test_data(self, Xtest, Ytest):
        """
        Parameters
        ----------
        Xtest : list
            Test images.
        Ytest : list
            Test masks.
        """
        self.Xtest = Xtest
        self.Ytest = Ytest

    def set_generator(self, generator=None, iterations=30):
        """
        Parameters
        ----------
        generator : GenLayer
            The generator layer.
        iterations : int
            Defines how long 1 epoch is. One iteration equals processing one batch.
        """
        self.generator = generator
        # noinspection PyAttributeOutsideInit
        self.iterations = iterations

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
            ExpField.path_to_arch: experiment[ExpField.path_to_arch],
            ExpField.pretrained_layers: experiment[ExpField.pretrained_layers],
            ExpField.weights: experiment[ExpField.weights],
            ExpField.untrainable_layers: experiment[ExpField.untrainable_layers],
            ExpField.epochs: experiment[ExpField.epochs],
            ExpField.test_period: experiment[ExpField.test_period],
            ExpField.class_names: experiment[ExpField.class_names],
            ExpField.save_period: experiment[ExpField.save_period],
            ExpField.loss_type: experiment[ExpField.loss_type],
            ExpField.l1_reg: experiment[ExpField.l1_reg],
            ExpField.l1_reg_layers: experiment[ExpField.l1_reg_layers],
            ExpField.l2_reg: experiment[ExpField.l2_reg],
            ExpField.l2_reg_layers: experiment[ExpField.l2_reg_layers]
        }
        for opt_info in experiment[ExpField.optimizers]:
            for b_sz in experiment[ExpField.batch_sizes]:
                for g in experiment[ExpField.gammas]:
                    exp_params[SubExpField.opt_info] = opt_info
                    exp_params[SubExpField.batch_sz] = b_sz
                    exp_params[SubExpField.gamma] = g
                    self._run_focal_experiment(exp_params)

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
        arch_path = exp_params[ExpField.path_to_arch]
        batch_sz = exp_params[SubExpField.batch_sz]
        if self.generator is None:
            model = Builder.segmentator_from_json(arch_path, batch_size=batch_sz)
        else:
            model = Builder.segmentator_from_json(arch_path, generator=self.generator)

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

    def _prepare_test_vars(self, model_name, exp_params):
        print('Preparing test variables...')
        # Create the test folder
        opt_name = exp_params[SubExpField.opt_info]['params']['name']
        gamma = exp_params[SubExpField.gamma]
        batch_size = exp_params[SubExpField.batch_sz]
        self.to_save_folder = os.path.join(
            self._exp_folder,
            f'{model_name}_gamma={gamma}_opt_name={opt_name}_bsz={batch_size}'
        )
        os.makedirs(self.to_save_folder, exist_ok=True)

        config_json = json.dumps(exp_params, indent=4)
        with open(f'{self.to_save_folder}/config.json', mode='w') as json_file:
            json_file.write(config_json)

        # Create test holders
        self.loss_list = []
        self.loss_info = {
            'loss_list': self.loss_list,
        }

        self.v_dice_test_list = []
        self.epochs_list = []
        self.test_info = {
            'epoch': self.epochs_list,
            'v_dice_list': self.v_dice_test_list
        }
        # Add sublists for each class
        self.dices_for_each_class = {}
        for class_name in exp_params[ExpField.class_names]:
            self.dices_for_each_class[class_name] = []
        self.test_info.update(self.dices_for_each_class)

# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------EXPERIMENT UTILITIES---------------------------------------------------------------

    def _perform_testing(self, model, exp_params, epoch):
        # COLLECT PREDICTIONS
        print('Testing the model...')
        print('Collecting predictions...')

        batch_sz = exp_params[SubExpField.batch_sz]
        Xtest, Ytest = shuffle(self.Xtest, self.Ytest)
        predictions = []
        labels = []
        n_batches = len(Xtest) // batch_sz
        for i in tqdm(range(n_batches)):
            labels += Ytest[i * batch_sz:(i + 1) * batch_sz]
            predictions += [model.predict(Xtest[i * batch_sz:(i + 1) * batch_sz])]
        predictions = np.vstack(predictions)
        labels = np.asarray(labels)

        print('Computing V-Dice...')

        # COMPUTE DICE AND CREATE CONFUSION MATRIX
        v_dice_val, dices = categorical_dice_coeff(predictions, labels, use_argmax=True)

        print('V-Dice:', v_dice_val)
        for i, class_name in enumerate(exp_params[ExpField.class_names]):
            self.dices_for_each_class[class_name] += [dices[i]]
            print(f'{class_name}:', dices[i])

        # Compute and save matrix
        conf_mat_path = self.to_save_folder + f'/mat_epoch={epoch}.png'
        print('Computing confusion matrix...')
        confusion_mat(
            predictions, labels, use_argmax_p=True, to_flatten=True,
            save_path=conf_mat_path, dpi=175
        )
        # Hope for freeing memory
        del labels
        del predictions

        print('Collecting data...')

        # COLLECT DATA
        self.epochs_list.append(epoch)
        self.v_dice_test_list.append(v_dice_val)

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------EXPERIMENT LOOP-------------------------------------------------------------------

    def _run_focal_experiment(self, exp_params):
        model = self._restore_model(exp_params)
        self._prepare_test_vars(model.name, exp_params)

        loss_type = exp_params[ExpField.loss_type]
        opt_info = exp_params[SubExpField.opt_info]
        gamma = exp_params[SubExpField.gamma]
        epochs = exp_params[ExpField.epochs]
        test_period = exp_params[ExpField.test_period]
        save_period = exp_params[ExpField.save_period]
        optimizer, global_step = OptimizerBuilder.build_optimizer(opt_info)
        if global_step is not None:
            self._sess.run(tf.variables_initializer([global_step]))
        # Catch InterruptException
        try:
            for i in range(epochs):
                if self.generator is None:
                    if loss_type == LossType.FocalLoss:
                        sub_train_info = model.fit_focal(
                            images=self.Xtrain, labels=self.Ytrain, gamma=gamma,
                            num_positives=self.num_pos, optimizer=optimizer, epochs=1, global_step=global_step
                        )
                    elif loss_type == LossType.MakiLoss:
                        sub_train_info = model.fit_maki(
                            images=self.Xtrain, labels=self.Ytrain, gamma=gamma,
                            num_positives=self.num_pos, optimizer=optimizer, epochs=1, global_step=global_step
                        )
                    elif loss_type == LossType.QuadraticCELoss:
                        sub_train_info = model.fit_quadratic_ce(
                            images=self.Xtrain, labels=self.Ytrain,
                            optimizer=optimizer, epochs=1, global_step=global_step
                        )
                    else:
                        raise ValueError('Unknown loss type!')
                else:
                    if loss_type == LossType.FocalLoss:
                        sub_train_info = model.genfit_focal(
                            gamma=gamma, optimizer=optimizer, epochs=1, iterations=self.iterations,
                            global_step=global_step
                        )
                    elif loss_type == LossType.MakiLoss:
                        sub_train_info = model.genfit_maki(
                            gamma=gamma, optimizer=optimizer, epochs=1,
                            iterations=self.iterations, global_step=global_step
                        )
                    elif loss_type == LossType.QuadraticCELoss:
                        sub_train_info = model.genfit_quadratic_ce(
                            optimizer=optimizer, epochs=1, iterations=self.iterations, global_step=global_step
                        )
                    else:
                        raise ValueError('Unknown loss type!')

                self.loss_list += sub_train_info['train losses']

                if i % test_period == 0:
                    self._perform_testing(model, exp_params, i)

                if save_period is not None and i % save_period == 0:
                    os.makedirs(
                        f'{self.to_save_folder}/epoch_{i}/', exist_ok=True
                    )
                    model.save_weights(f'{self.to_save_folder}/epoch_{i}/weights.ckpt')
                print('Epochs:', i)
        except KeyboardInterrupt as ex:
            traceback.print_exc()
            print("SAVING GAINED DATA")
        finally:
            # ALWAYS DO LAST SAVE
            os.makedirs(
                f'{self.to_save_folder}/last_weights/', exist_ok=True
            )
            model.save_weights(f'{self.to_save_folder}/last_weights/weights.ckpt')
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

            self._save_test_info()
            self._create_dice_loss_graphs()
            print('Sub test is done.')

# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------SAVING TRAINING RESULTS------------------------------------------------------------

    def _save_test_info(self):
        test_df = pd.DataFrame(self.test_info)
        test_df.to_csv(f'{self.to_save_folder}/test_info.csv')
        loss_df = pd.DataFrame(self.loss_info)
        loss_df.to_csv(f'{self.to_save_folder}/loss_info.csv')

    def _create_dice_loss_graphs(self):
        labels = [key for key in self.test_info]
        values = [self.test_info[key] for key in self.test_info]
        # Plot all dices
        TestVisualizer.plot_test_values(
            test_values=values[1:],
            legends=labels[1:],
            x_label='Epochs',
            y_label='Dice',
            save_path=f'{self.to_save_folder}/dices.png'
        )
        # Plot V-Dice
        TestVisualizer.plot_test_values(
            test_values=values[1:2],
            legends=labels[1:2],
            x_label='Epochs',
            y_label='V-Dice',
            save_path=f'{self.to_save_folder}/v_dice.png'
        )
        # Plot Loss
        losses = [self.loss_info[key] for key in self.loss_info]
        loss_names = [key for key in self.loss_info]

        TestVisualizer.plot_test_values(
            test_values=losses,
            legends=loss_names,
            x_label='Epochs',
            y_label='Loss',
            save_path=f'{self.to_save_folder}/loss.png'
        )
        # Normalize loss and V-Dice and then plot
        normalized_losses = []
        for loss in losses:
            # Take loss values at certain epochs when the model was tested
            np_loss = np.array(loss)[values[0]]
            std = np_loss.std()
            mean = np_loss.mean()
            normed_loss = (np_loss - mean) / std
            normalized_losses += [normed_loss]

        np_v_dice = np.array(values[1])
        std = np_v_dice.std()
        mean = np_v_dice.mean()
        normed_v_dice = (np_v_dice - mean) / std
        TestVisualizer.plot_test_values(
            test_values=normalized_losses + [normed_v_dice],
            legends=loss_names + ['V-Dice'],
            x_label='Epochs',
            y_label='Norm',
            save_path=f'{self.to_save_folder}/norm.png'
        )
