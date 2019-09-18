from __future__ import absolute_import
import json
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from makiflow.metrics import categorical_dice_coeff, confusion_mat
from makiflow.trainers.optimizer_builder import OptimizerBuilder
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
    'utrainable layers': [layer_name]
}"""


# SEGMENTATOR IMPLIES THAT ALL NETWORK ARCHITECTURES HAVE THE SAME INPUT SHAPE
class SegmentatorTrainer:
    def __init__(self, exp_params, path_to_save: str):
        self._exp_params = exp_params
        if isinstance(exp_params, str):
            self._exp_params = self._load_exp_params(exp_params)
        self._path_to_save = path_to_save
        self._sess = None

    def _load_exp_params(self, json_path):
        with open(json_path) as json_file:
            json_value = json_file.read()
            exp_params = json.loads(json_value)
        return exp_params

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------SETTING UP THE EXPERIMENTS-----------------------------------------------------------

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
        self._create_experiment_folder(experiment['name'])
        exp_params = {
            'name': experiment['name'],
            'path_to_arch': experiment['path to arch'],
            'pretrained_layers': experiment['pretrained layers'],
            'weights': experiment['weights'],
            'utrainable_layers': experiment['untrainable layers'],
            'epochs': experiment['epochs'],
            'test_period': experiment['test period'],
            'class_names': experiment['class names'],
            'save_period': experiment['save period'],
            'loss_type': experiment['loss type']
        }
        for opt_info in experiment['optimizers']:
            for b_sz in experiment['batch sizes']:
                for g in experiment['gammas']:
                    exp_params['opt_info'] = opt_info
                    exp_params['batch_sz'] = b_sz
                    exp_params['gamma'] = g
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
        arch_path = exp_params['path_to_arch']
        batch_sz = exp_params['batch_sz']
        model = Builder.segmentator_from_json(arch_path, batch_size=batch_sz)

        weights_path = exp_params['weights']
        pretrained_layers = exp_params['pretrained_layers']
        untrainable_layers = exp_params['utrainable_layers']

        model.set_session(self._sess)
        if weights_path is not None:
            model.load_weights(weights_path, layer_names=pretrained_layers)

        if untrainable_layers is not None:
            layers = []
            for layer_name in untrainable_layers:
                layers += [(layer_name, False)]
            model.set_layers_trainable(layers)
        return model

    def _prepare_test_vars(self, model_name, exp_params):
        print('Preparing test variables...')
        # Create the test folder
        opt_name = exp_params['opt_info']['params']['name']
        gamma = exp_params['gamma']
        batch_size = exp_params['batch_sz']
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
        for class_name in exp_params['class_names']:
            self.dices_for_each_class[class_name] = []
        self.test_info.update(self.dices_for_each_class)

# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------EXPERIMENT UTILITIES---------------------------------------------------------------

    def _perform_testing(self, model, exp_params, epoch):
        # COLLECT PREDICTIONS
        print('Testing the model...')
        print('Collecting predictions...')

        batch_sz = exp_params['batch_sz']
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
        print('V-Dice:', v_dice_val)
        for i, class_name in enumerate(exp_params['class_names']):
            self.dices_for_each_class[class_name] += [dices[i]]
            print(f'{class_name}:', dices[i])

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------EXPERIMENT LOOP-------------------------------------------------------------------

    def _run_focal_experiment(self, exp_params):
        model = self._restore_model(exp_params)
        self._prepare_test_vars(model.name, exp_params)

        loss_type = exp_params['loss_type']
        opt_info = exp_params['opt_info']
        gamma = exp_params['gamma']
        epochs = exp_params['epochs']
        test_period = exp_params['test_period']
        save_period = exp_params['save_period']
        optimizer = OptimizerBuilder.build_optimizer(opt_info)
        # Catch InterruptException
        try:
            for i in range(epochs):
                if loss_type == 'FocalLoss':
                    sub_train_info = model.fit_focal(
                        images=self.Xtrain, labels=self.Ytrain, gamma=gamma,
                        num_positives=self.num_pos, optimizer=optimizer, epochs=1
                    )
                elif loss_type == 'MakiLoss':
                    sub_train_info = model.fit_maki(
                        images=self.Xtrain, labels=self.Ytrain, gamma=gamma,
                        num_positives=self.num_pos, optimizer=optimizer, epochs=1
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

            self._save_test_info()
            self._create_dice_loss_graphs()
            print('Sub test is done')

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
