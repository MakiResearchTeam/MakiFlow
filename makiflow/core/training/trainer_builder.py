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

from .core.aion import Aion


class TrainerBuilder:
    TYPE = Aion.TYPE
    PARAMS = Aion.PARAMS

    # Contains pairs
    TRAINERS = {}

    @staticmethod
    def register_trainer(trainer_class):
        TrainerBuilder.TRAINERS.update(
            {trainer_class.TYPE: trainer_class}
        )

    @staticmethod
    def trainer_from_dict(model, train_inputs, label_tensors, info_dict):
        trainer_type = info_dict[TrainerBuilder.TYPE]
        params = info_dict[TrainerBuilder.PARAMS]
        trainer_class = TrainerBuilder.TRAINERS.get(trainer_type)
        assert trainer_type is not None, f'There is no trainer with TYPE={trainer_type}'
        trainer_object = trainer_class(
            model=model,
            train_inputs=train_inputs,
            label_tensors=label_tensors
        )
        trainer_object.set_params(params)
        return trainer_object

