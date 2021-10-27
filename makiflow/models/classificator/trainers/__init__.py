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

from .maki_loss import MakiLossTrainer
from .focal_loss import FocalTrainer
from .ce_loss import CETrainer
from .qce_loss import QCETrainer
from .weighted_ce_loss import WeightedCETrainer
from .dice_loss import DiceTrainer
from .focal_binary_loss import FocalBinaryTrainer
from .focal_loss_w_weight_mask_as_last_mask import FocalLossWweightsMaskAsLastMaskTrainer

