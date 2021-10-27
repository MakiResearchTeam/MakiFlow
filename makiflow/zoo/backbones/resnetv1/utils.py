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

WITHOUT_POINTWISE = 'without_pointwise'
WITH_POINTWISE = 'with_pointwise'


def get_batchnorm_params():
    return {
            'decay': 0.9,
            'eps': 1e-3
    }


def get_batchnorm_params_resnet34():
    return {
            'decay': 0.99,
            'eps': 2e-5
    }


def get_head_batchnorm_params():
    return {
            'use_gamma': False,
            'decay': 0.9,
            'eps': 1e-3
    }
