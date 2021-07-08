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

import numpy as np
from abc import abstractmethod, ABC

# Some initialize methods
# Initializations define the way to set the initial random weights of MakiFlow layers.

# (in_f, out_f), in most cases
NUM_OF_PARAMS_DENSE = 2


class BaseInitializer(ABC):
    Name = 'Base'

    def __str__(self):
        return self.Name

    @abstractmethod
    def __call__(self, shape: list, dtype=np.float32):
        pass

    def _check_input_shape(self, shape: list) -> list:
        # Check size and type
        if len(shape) == 0 or len(shape) == 1 or (not isinstance(shape, list) and not isinstance(shape, tuple)):
            raise ValueError(f"Input shape must be list or tuple "
                             f"\nand must have len more than 1, but len {len(shape)} was given.")
        shape = list(shape)
        return shape


class XavierGaussianAvg(BaseInitializer):

    Name = 'XavierGaussianAvg'

    def __call__(self, shape: list, dtype=np.float32):
        """
        Generate array according to input `shape` with certain initialization
        In this case its Xavier Gaussian Avg

        Parameters
        ----------
        shape : list or tuple
            Shape of the final matrix, must be with length more than 1
        dtype : np.dtype
            Type of the final matrix

        Returns
        -------
        np.float32
            Final matrix according to class name of the initialization

        """
        shape = super()._check_input_shape(shape=shape)

        w = np.random.randn(*shape)
        if len(shape) == NUM_OF_PARAMS_DENSE:
            # Dense like, (out_f, in_f))
            w *= np.sqrt( 2.0 / (shape[-1] + shape[-2]))
        else:
            # Conv like or more dimensions
            sum_div = 0.0
            one_part_mul = 1
            # kw * kh * ...
            # Except out feature and input, multiply size of overall kernel
            for elem_k in shape[:-2]:
                one_part_mul *= elem_k
            # Multiply on in/out feature
            for elem_f in shape[-2:]:
                sum_div += one_part_mul * elem_f
            # Divide
            w *= np.sqrt(2. / sum_div)
        return w.astype(dtype)


class XavierGaussianInf(BaseInitializer):

    Name = 'XavierGaussianInf'

    def __call__(self, shape: list, dtype=np.float32):
        """
        Generate array according to input `shape` with certain initialization
        In this case its Xavier Gaussian Inf

        Parameters
        ----------
        shape : list or tuple
            Shape of the final matrix, must be with length more than 1
        dtype : np.dtype
            Type of the final matrix

        Returns
        -------
        np.float32
            Final matrix according to class name of the initialization

        """
        shape = super()._check_input_shape(shape=shape)

        w = np.random.randn(*shape)
        if len(shape) == NUM_OF_PARAMS_DENSE:
            # Dense like, (out_f, in_f))
            # Divide only fot in_f
            w *= np.sqrt(1.0 / (shape[-1]))
        else:
            # Conv like or more dimensions
            one_part_mul = 1
            # kw * kh * ...
            # Except out feature and input, multiply size of overall kernel
            for elem_k in shape[:-2]:
                one_part_mul *= elem_k
            # Multiply on number of the in features
            sum_div = one_part_mul * shape[-1]
            # Divide
            w *= np.sqrt(1. / sum_div)
        return w.astype(dtype)


class XavierUniformAvg(BaseInitializer):

    Name = 'XavierUniformAvg'

    def __call__(self, shape: list, dtype=np.float32):
        """
        Generate array according to input `shape` with certain initialization
        In this case its Xavier Uniform Avg

        Parameters
        ----------
        shape : list or tuple
            Shape of the final matrix, must be with length more than 1
        dtype : np.dtype
            Type of the final matrix

        Returns
        -------
        np.float32
            Final matrix according to class name of the initialization

        """
        shape = super()._check_input_shape(shape=shape)

        w = np.random.uniform(low=-1.0, high=1.0, size=shape)
        if len(shape) == NUM_OF_PARAMS_DENSE:
            # Dense like, (out_f, in_f))
            w *= np.sqrt(6. / (shape[-1] + shape[-2]))
        else:
            # Conv like or more dimensions
            sum_div = 0.0
            one_part_mul = 1
            # kw * kh * ...
            # Except out feature and input, multiply size of overall kernel
            for elem_k in shape[:-2]:
                one_part_mul *= elem_k
            # Multiply on in/out feature
            for elem_f in shape[-2:]:
                sum_div += one_part_mul * elem_f
            # Divide
            w *= np.sqrt(6. / sum_div)
        return w.astype(dtype)


class XavierUniformInf(BaseInitializer):

    Name = 'XavierUniformInf'

    def __call__(self, shape: list, dtype=np.float32):
        """
        Generate array according to input `shape` with certain initialization
        In this case its Xavier Uniform Inf

        Parameters
        ----------
        shape : list or tuple
            Shape of the final matrix, must be with length more than 1
        dtype : np.dtype
            Type of the final matrix

        Returns
        -------
        np.float32
            Final matrix according to class name of the initialization

        """
        shape = super()._check_input_shape(shape=shape)

        w = np.random.uniform(low=-1.0, high=1.0, size=shape)
        if len(shape) == NUM_OF_PARAMS_DENSE:
            # Dense like, (out_f, in_f))
            # Divide only fot in_f
            w *= np.sqrt(3. / (shape[-1]))
        else:
            # Conv like or more dimensions
            one_part_mul = 1
            # kw * kh * ...
            # Except out feature and input, multiply size of overall kernel
            for elem_k in shape[:-2]:
                one_part_mul *= elem_k
            # Multiply on number of the in features
            sum_div = one_part_mul * shape[-1]
            # Divide
            w *= np.sqrt(3. / sum_div)
        return w.astype(dtype)


class Lasange(BaseInitializer):

    Name = 'Lasange'

    def __call__(self, shape: list, dtype=np.float32):
        """
        Generate array according to input `shape` with certain initialization
        In this case its Lasange

        Parameters
        ----------
        shape : list or tuple
            Shape of the final matrix, must be with length more than 1
        dtype : np.dtype
            Type of the final matrix

        Returns
        -------
        np.float32
            Final matrix according to class name of the initialization

        """
        shape = super()._check_input_shape(shape=shape)

        w = np.random.randn(*shape)
        if len(shape) == NUM_OF_PARAMS_DENSE:
            # Dense like, (out_f, in_f))
            w *= np.sqrt(12.0 / (shape[-1] + shape[-2]))
        else:
            # Conv like or more dimensions
            sum_div = 0.0
            one_part_mul = 1
            # kw * kh * ...
            # Except out feature and input, multiply size of overall kernel
            for elem_k in shape[:-2]:
                one_part_mul *= elem_k
            # Multiply on in/out feature
            for elem_f in shape[-2:]:
                sum_div += one_part_mul * elem_f
            # Divide
            w *= np.sqrt(12. / sum_div)
        return w.astype(dtype)


class He(BaseInitializer):

    Name = 'He'

    def __call__(self, shape: list, dtype=np.float32):
        """
        Generate array according to input `shape` with certain initialization
        In this case its He

        Parameters
        ----------
        shape : list or tuple
            Shape of the final matrix, must be with length more than 1
        dtype : np.dtype
            Type of the final matrix

        Returns
        -------
        np.float32
            Final matrix according to class name of the initialization

        """
        shape = super()._check_input_shape(shape=shape)

        w = np.random.randn(*shape)
        if len(shape) == NUM_OF_PARAMS_DENSE:
            # Dense like, (out_f, in_f))
            # Divide only fot in_f
            w *= np.sqrt(2. / (shape[-1]))
        else:
            # Conv like or more dimensions
            one_part_mul = 1
            # kw * kh * ...
            # Except out feature and input, multiply size of overall kernel
            for elem_k in shape[:-2]:
                one_part_mul *= elem_k
            # Multiply on number of the in features
            sum_div = one_part_mul * shape[-1]
            # Divide
            w *= np.sqrt(2. / sum_div)
        return w.astype(dtype)


class HeGrad(BaseInitializer):

    Name = 'HeGrad'

    def __call__(self, shape: list, dtype=np.float32):
        """
        Generate array according to input `shape` with certain initialization
        In this case its He Grad

        Parameters
        ----------
        shape : list or tuple
            Shape of the final matrix, must be with length more than 1
        dtype : np.dtype
            Type of the final matrix

        Returns
        -------
        np.float32
            Final matrix according to class name of the initialization

        """
        shape = super()._check_input_shape(shape=shape)

        w = np.random.randn(*shape)
        if len(shape) == NUM_OF_PARAMS_DENSE:
            # Dense like, (out_f, in_f))
            # Divide only fot out_f
            w *= np.sqrt(2. / (shape[-2]))
        else:
            # Conv like or more dimensions
            one_part_mul = 1
            # kw * kh * ...
            # Except out feature and input, multiply size of overall kernel
            for elem_k in shape[:-2]:
                one_part_mul *= elem_k
            # Multiply on number of the in features
            sum_div = one_part_mul * shape[-2]
            # Divide
            w *= np.sqrt(2. / sum_div)
        return w.astype(dtype)


class HeUniform(BaseInitializer):

    Name = 'HeUniform'

    def __call__(self, shape: list, dtype=np.float32):
        """
        Generate array according to input `shape` with certain initialization
        In this case its He Uniform

        Parameters
        ----------
        shape : list or tuple
            Shape of the final matrix, must be with length more than 1
        dtype : np.dtype
            Type of the final matrix

        Returns
        -------
        np.float32
            Final matrix according to class name of the initialization

        """
        shape = super()._check_input_shape(shape=shape)

        w = np.random.uniform(low=-1.0, high=1.0, size=shape)
        if len(shape) == NUM_OF_PARAMS_DENSE:
            # Dense like, (out_f, in_f))
            # Divide only fot in_f
            w *= np.sqrt(2. / (shape[-1]))
        else:
            # Conv like or more dimensions
            one_part_mul = 1
            # kw * kh * ...
            # Except out feature and input, multiply size of overall kernel
            for elem_k in shape[:-2]:
                one_part_mul *= elem_k
            # Multiply on number of the in features
            sum_div = one_part_mul * shape[-1]
            # Divide
            w *= np.sqrt(2. / sum_div)
        return w.astype(dtype)


class RandomNormal(BaseInitializer):

    Name = 'RandomNormal'

    def __init__(self, mean=0.0, variance=0.01):
        self._mean = mean
        self._variance = variance

    def __call__(self, shape: list, dtype=np.float32):
        """
        Generate array according to input `shape` with certain initialization
        In this case its Random Normal

        Parameters
        ----------
        shape : list or tuple
            Shape of the final matrix, must be with length more than 1
        dtype : np.dtype
            Type of the final matrix

        Returns
        -------
        np.float32
            Final matrix according to class name of the initialization

        """
        shape = super()._check_input_shape(shape=shape)

        w = np.random.normal(loc=self._mean, scale=self._variance, size=shape)
        return w.astype(dtype)


class InitController:

    SET_INITS = {
        XavierGaussianAvg.Name: XavierGaussianAvg,
        XavierGaussianInf.Name: XavierGaussianInf,

        XavierUniformAvg.Name:  XavierUniformAvg,
        XavierUniformInf.Name:  XavierUniformInf,

        Lasange.Name:           Lasange,

        He.Name:                He,
        HeGrad.Name:            HeGrad,
        HeUniform.Name:         HeUniform,

        RandomNormal.Name:      RandomNormal,
    }

