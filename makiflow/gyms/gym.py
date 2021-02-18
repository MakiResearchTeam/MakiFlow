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

import json
import os
import tensorflow as tf
from makiflow.gyms.utils.optimizer_builder import OptimizerBuilder
from makiflow.gyms.core import TesterBase
from makiflow.gyms.gyms_modules.gyms_collector import GYM_COLLECTOR, ASSEMBLER, TESTER


class Gym:
    MSG_CREATE_FOLDER = 'Creating gym folder...'
    WRN_GYM_FOLDER_EXISTS = 'Warning! Gym folder with the specified path={0} already' \
                            'exists. Creating another directory with path={1}.'
    """
    Config file consists of several main sections:
    - model_config
    - trainer_config
    - training_config
    - testing_config
    - tb_config
    - distillation config
    """
    TYPE = 'type'
    TRAIN_CONFIG = 'training_config'
    EPOCHS = 'epochs'
    ITERS = 'iters'
    TEST_PERIOD = 'test_period'
    SAVE_PERIOD = 'save_period'
    PRINT_PERIOD = 'print_period'
    GYM_FOLDER = 'gym_folder'
    OPTIMIZER_INFO = 'optimizer_info'
    GEN_LAYER_INFO = "genlayer_config"
    SIZE_IMG = "im_hw"

    TB_CONFIG = 'tb_config'
    LAYER_HISTS = 'layer_histograms'

    def __init__(self, config_path, gen_layer_fabric, sess):
        """
        Training gym for a pose estimation model.
        Parameters
        ----------
        config_path : str
            Path to the config file.
        gen_layer_fabric : python function
            Fabric that create a generator layer given the arguments.
        sess : tf.Session
            The session object.
        """
        with open(config_path) as json_file:
            json_value = json_file.read()
            config = json.loads(json_value)
        self._type = config[Gym.TYPE]
        self._train_config = config[Gym.TRAIN_CONFIG]
        self._gen_layer_fabric = gen_layer_fabric
        self._sess = sess

        self._setup_gym(config)
        self._setup_tensorboard(config)

    def _setup_gym(self, config):
        self._create_gym_folder()

        # Create folder for the last weights of the model
        self._last_w_folder_path = os.path.join(
            self._train_config[Gym.GYM_FOLDER],
            'last_weights'
        )
        os.makedirs(self._last_w_folder_path, exist_ok=True)

        # Create folder for the tensorboard and create tester
        tensorboard_path = os.path.join(
            self._train_config[Gym.GYM_FOLDER],
            'tensorboard'
        )
        os.makedirs(tensorboard_path, exist_ok=True)
        config[TesterBase.TB_FOLDER] = tensorboard_path
        self._tb_path = tensorboard_path
        self._tester = GYM_COLLECTOR[self._type][TESTER](
            config,
            self._sess,
            self._train_config[Gym.GYM_FOLDER]
        )

        # Create model, trainer and set the tensorboard folder
        self._trainer, self._model = GYM_COLLECTOR[self._type][ASSEMBLER].assemble(
            config,
            self._gen_layer_fabric,
            self._sess
        )
        self._hermes = self._trainer.get_hermes()
        self._hermes.set_tensorboard_writer(self._tester.get_writer())

    def _create_gym_folder(self):
        print(Gym.MSG_CREATE_FOLDER)
        orig_path = self._train_config[Gym.GYM_FOLDER]
        path = orig_path
        counter = 1
        while os.path.isdir(path):
            new_path = orig_path + f'_{counter}'
            counter += 1
            print(Gym.WRN_GYM_FOLDER_EXISTS.format(path, new_path))
            path = new_path

        self._train_config[Gym.GYM_FOLDER] = path
        os.makedirs(path, exist_ok=True)

    def _setup_tensorboard(self, config):
        print('Configuring tensorboard histograms...')
        tb_config = config.get(Gym.TB_CONFIG)
        if tb_config is None:
            print('No config for tensorboard. Skipping the step.')
            return

        layer_names = tb_config.get(Gym.LAYER_HISTS)
        if layer_names is None:
            layer_names = []
        self._hermes.set_layers_histograms(layer_names)

    def get_tb_path(self):
        return self._tb_path

    def get_model(self):
        """
        May be used for adding a custom loss to the model.
        """
        return self._model

    def start_training(self):
        self._save_arch()
        epochs = self._train_config[Gym.EPOCHS]
        iters = self._train_config[Gym.ITERS]
        test_period = self._train_config[Gym.TEST_PERIOD]
        save_period = self._train_config[Gym.SAVE_PERIOD]
        print_period = self._train_config[Gym.PRINT_PERIOD]

        optimizer, global_step = OptimizerBuilder.build_optimizer(
            self._train_config[Gym.OPTIMIZER_INFO]
        )
        
        if global_step is not None:
            self._sess.run(tf.variables_initializer([global_step]))

        it_counter = 0
        for i in range(1, epochs + 1):
            _ = self._trainer.fit(
                optimizer=optimizer, epochs=1, iter=iters, global_step=global_step, print_period=print_period
            )
            it_counter += iters

            if i % test_period == 0 or i % save_period == 0:
                path_save_res = self._create_new_gym_folder(i)
                if i % test_period == 0:
                    self._tester.evaluate(self._model, it_counter, path_save_res)
                if i % save_period == 0:
                    self._save_weights(path_save_res)

        self._tester.final_eval(self._last_w_folder_path)
        path_to_save = os.path.join(
            self._last_w_folder_path, 'weights.ckpt'
        )
        self._model.save_weights(path_to_save)

    def _create_new_gym_folder(self, epoch: int) -> str:
        gym_folder = self._train_config[Gym.GYM_FOLDER]
        save_path = os.path.join(
            gym_folder, f'epoch_{epoch}'
        )
        os.makedirs(save_path, exist_ok=True)

        return save_path

    def _save_weights(self, save_path: str):
        save_path = os.path.join(
            save_path, 'weights.ckpt'
        )
        self._model.save_weights(save_path)

    def _save_arch(self):
        gym_folder = self._train_config[Gym.GYM_FOLDER]
        save_path = os.path.join(
            gym_folder, 'model.json'
        )
        self._model.save_architecture(save_path)
