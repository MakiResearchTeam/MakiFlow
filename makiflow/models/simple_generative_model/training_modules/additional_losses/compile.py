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

from .perceptual_loss import PerceptualLossModuleGenerator
from makiflow.base.maki_entities import MakiTensor


class BasicTrainingModule(PerceptualLossModuleGenerator):
    """
    Connect additional losses

    """

    def __init__(self,
                 input_x: MakiTensor,
                 output_x: MakiTensor,
                 name="SimpleGenerativeModel"
    ):
        self._perceptual_loss_vars_are_ready = False
        super().__init__(input_x=input_x,
                         output_x=output_x,
                         name=name
        )

    def _build_additional_losses(self, total_loss):
        if super().is_use_perceptual_loss():
            total_loss += self._build_perceptual_loss()

        return total_loss

