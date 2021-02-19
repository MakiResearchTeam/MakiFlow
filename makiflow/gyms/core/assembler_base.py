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

from makiflow.models.classificator import Classificator
from makiflow.core import TrainerBuilder
from makiflow.layers import InputLayer
from makiflow.distillation.core import DistillatorBuilder
from makiflow.tools.converter import to_makitensor


class ModelAssemblerBase:
    # model config
    MODEL_CONFIG = 'model_config'
    ARCH_PATH = 'arch_path'
    WEIGHTS_PATH = 'weights_path'
    PRETRAINED_LAYERS = 'pretrained_layers'

    # Trainer config
    TRAINER_CONFIG = 'trainer_config'
    TRAINER_INFO = 'trainer_info'
    L1_REG = 'l1_reg'
    L1_REG_LAYERS = 'l1_reg_layers'
    L2_REG = 'l2_reg'
    L2_REG_LAYERS = 'l2_reg_layers'
    UNTRAINABLE_LAYERS = 'untrainable_layers'
    DISTILLATION = 'distillation_info'
    # Distillation info
    TEACHER_WEIGHTS = 'weights'
    TEACHER_ARCH = 'arch'

    # gen_layer config
    GENLAYER_CONFIG = 'genlayer_config'
    DATA_GEN_PATH = 'data_gen_path'
    IM_HW = 'im_hw'
    PREFETCH_SIZE = 'prefetch_size'
    BATCH_SZ = 'batch_size'

    @staticmethod
    def assemble(config, gen_layer_fabric, sess):
        gen_layer = ModelAssemblerBase.build_gen_layer(config[ModelAssemblerBase.GENLAYER_CONFIG], gen_layer_fabric)
        model, type_model = ModelAssemblerBase.setup_model(config[ModelAssemblerBase.MODEL_CONFIG], gen_layer, sess)
        trainer = ModelAssemblerBase.setup_trainer(
            config[ModelAssemblerBase.TRAINER_CONFIG],
            model=model,
            type_model=type_model,
            gen_layer=gen_layer
        )
        return trainer, model

    @staticmethod
    def build_gen_layer(config, gen_layer_fabric):
        return gen_layer_fabric(
            data_gen_path=config[ModelAssemblerBase.DATA_GEN_PATH],
            im_hw=config[ModelAssemblerBase.IM_HW],
            batch_sz=config[ModelAssemblerBase.BATCH_SZ],
            prefetch_sz=config[ModelAssemblerBase.PREFETCH_SIZE],
        )

    @staticmethod
    def setup_model(model_config, gen_layer, sess):
        shape = gen_layer.get_shape()
        # Change batch_size to 1
        shape[0] = 1
        # Change image size to dynamic size
        shape[1] = None
        shape[2] = None
        name = gen_layer.get_name()

        input_layer = InputLayer(input_shape=shape, name=name)
        # TODO: Watcher for type of model????
        type_model = Classificator
        model = type_model.from_json(model_config[ModelAssemblerBase.ARCH_PATH], input_layer)
        model.set_session(sess)

        # Load pretrained weights
        weights_path = model_config[ModelAssemblerBase.WEIGHTS_PATH]
        pretrained_layers = model_config[ModelAssemblerBase.PRETRAINED_LAYERS]
        if weights_path is not None:
            model.load_weights(weights_path, layer_names=pretrained_layers)

        return model, type_model

    @staticmethod
    def _setup_trainer(trainer, config_data: dict, model, type_model):
        untrainable_layers = config_data[ModelAssemblerBase.UNTRAINABLE_LAYERS]
        if untrainable_layers is not None:
            layers = []
            for layer_name in untrainable_layers:
                layers += [(layer_name, False)]
            trainer.set_layers_trainable(layers)

        # Set l1 regularization
        l1_reg = config_data[ModelAssemblerBase.L1_REG]
        if l1_reg is not None:
            l1_reg = float(l1_reg)
            l1_reg_layers = config_data[ModelAssemblerBase.L1_REG_LAYERS]
            reg_config = [(layer, l1_reg) for layer in l1_reg_layers]
            trainer.set_l1_reg(reg_config)

        # Set l2 regularization
        l2_reg = config_data[ModelAssemblerBase.L2_REG]
        if l2_reg is not None:
            l2_reg = float(l2_reg)
            l2_reg_layers = config_data[ModelAssemblerBase.L2_REG_LAYERS]
            reg_config = [(layer, l2_reg) for layer in l2_reg_layers]
            trainer.set_l2_reg(reg_config)

        distillation_config = config_data.get(ModelAssemblerBase.DISTILLATION)
        if distillation_config is not None:
            arch_path = distillation_config[ModelAssemblerBase.TEACHER_ARCH]
            teacher = type_model.from_json(arch_path)
            teacher.set_session(model.get_session())

            weights_path = distillation_config[ModelAssemblerBase.TEACHER_WEIGHTS]
            teacher.load_weights(weights_path)

            distillator = DistillatorBuilder.distillator_from_dict(
                teacher=teacher,
                info_dict=distillation_config
            )
            trainer = distillator(trainer)

        trainer.compile()
        return trainer


