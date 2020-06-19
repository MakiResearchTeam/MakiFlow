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


from makiflow.models.simple_generative_model.main_modules import SimpleGenerativeModelBasic


class PerceptualLossModuleGenerator(SimpleGenerativeModelBasic):

    NOTIFY_BUILD_PERCEPTUAL_LOSS = "Perceptual loss was built"

    def _prepare_training_vars_additional_losses(self):
        self._scale_per_loss = 1.0
        self._use_perceptual_loss = False
        self._creation_per_loss = None
        self._perceptual_loss_is_built = False

        self._perceptual_loss_vars_are_ready = True

    def is_use_perceptual_loss(self) -> bool:
        """
        Return bool variable which shows whether it is being used perceptual loss or not.
        """
        if not self._perceptual_loss_vars_are_ready:
            return self._perceptual_loss_vars_are_ready

        return self._use_perceptual_loss

    def add_perceptual_loss(self, creation_per_loss, scale_loss=1e-2):
        """
        Add the function that create percetual loss inplace.
        Parameters
        ----------
        creation_per_loss : func
            Function which will create percetual loss.
            This function must have 3 main input: input_image, target_image, sess.
            Example of function:
                def create_loss(input_image, target_image, sess):
                    ...
                    ...
                    return percetual_loss
            Where percetual_loss - is tensorflow Tensor
        scale_loss : float
            Scale of the perceptual loss.
        """
        if not self._perceptual_loss_vars_are_ready:
            PerceptualLossModuleGenerator._prepare_training_vars_additional_losses(self)
        self._creation_per_loss = creation_per_loss
        self._scale_per_loss = scale_loss
        self._use_perceptual_loss = True

    def _build_perceptual_loss(self):
        if not self._perceptual_loss_is_built:
            self._perceptual_loss = self._creation_per_loss(
                self._input_images,
                self._target_images,
                self._session
            ) * self._scale_per_loss

            print(PerceptualLossModuleGenerator.NOTIFY_BUILD_PERCEPTUAL_LOSS)
            self._perceptual_loss_is_built = True

        return self._perceptual_loss

