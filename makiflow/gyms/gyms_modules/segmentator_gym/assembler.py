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
from makiflow.core import TrainerBuilder
from makiflow.gyms.core.assembler_base import ModelAssemblerBase


class ModelAssemblerSegmentator(ModelAssemblerBase):

    @staticmethod
    def setup_trainer(config_data: dict, model, type_model, gen_layer):
        iterator = gen_layer.get_iterator()
        # TODO: Label tensor - tensors from iterator - how connect different models???
        trainer = TrainerBuilder.trainer_from_dict(
            model=model,
            train_inputs=[gen_layer],
            label_tensors={
                "LABELS": iterator['mask'],
                "WEIGHT_MAP": None
            },
            info_dict=config_data[ModelAssemblerBase.TRAINER_INFO]
        )
        super()._setup_trainer(trainer, config_data, model, type_model)

