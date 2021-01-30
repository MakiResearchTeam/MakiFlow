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

from .model_fitter import ModelFitter
from .gradient_variables_watcher import GradientVariablesWatcher
from .tensorboard import TensorBoard
from .loss_builder import Loss
from .trainer_builder import TrainerBuilder
from abc import ABC


class MakiTrainer(ModelFitter, ABC):
    pass


del ModelFitter
del ABC


"""
MakiTrainer consists of several layers of abstraction:

^------------------------------------MakiTrainer-----------------------------------^
| Dummy wrapper.                                                                   |
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

|------------------------------------ModelFitter-----------------------------------|
| Contains fit loops and API for loss tracking.                                    |
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

^-----------------------------------TrainingCore-----------------------------------^
| Dummy wrapper.                                                                   |
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

|------------------------------L2RegularizationModule------------------------------|
| Contains API for adding L2 regularization to the loss.                           |
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

^------------------------------L1RegularizationModule------------------------------^
| Contains API for adding L1 regularization to the loss.                           |
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

|------------------------------------Serializer------------------------------------|
| Contains API for serializing the model: useful tools, architecture saving,       |
| weights saving.                                                                  |
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

^--------------------------------TrainGraphCompiler--------------------------------^
| Compiles training graph and provides and API to access it.                       |
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
"""
