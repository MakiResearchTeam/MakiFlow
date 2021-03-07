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

import tensorflow as tf

"""
{
    "type": "ExponentialDecay",
    "params": {
        "learning_rate": ..
        "decay_steps": ..
    }
}
"""


class LearningRateBuilder:

    TYPE_FIELD = 'type'
    PARAMS = 'params'
    LEARNING_RATE = 'learning_rate'
    NAME = 'name'

    EXPONENTIAL_DECAY = 'ExponentialDecay'
    DECAY_STEPS = 'decay_steps'
    DECAY_RATE = 'decay_rate'
    STAIRCASE = 'staircase'

    COSINE_DECAY = 'CosineDecay'
    ALPHA = 'alpha'

    COSINE_RESTARTS_DECAY = 'CosineRestartsDecay'
    T_MUL = 't_mul'
    M_MUL = 'm_mul'

    INVERSE_TIME_DECAY = 'InverseTimeDecay'
    LINEAR_COSINE_DECAY = 'LinearCosineDecay'
    NUM_PERIODS = 'num_periods'
    BETA = 'beta'

    NATURAL_EXP_DECAY = 'NaturalExpDecay'

    NOISE_LINEAR_COSINE_DECAY = 'NoiseLinearCosineDecay'
    INITIAL_VARIANCE = 'initial_variance'
    VARIANCE_DECAY = 'variance_decay'

    PIECEWISE_CONSTANT_DECAY = 'PiecewiseConstantDecay'
    BOUNDARIES = 'boundaries'
    VALUES = 'values'

    POLYNOMIAL_DECAY = 'PolynomialDecay'
    END_LEARNING_RATE = 'end_learning_rate'
    POWER = 'power'
    CYCLE = 'cycle'

    @staticmethod
    def build_learning_rate(learning_rate_info):
        """
        Build learning rate with certain params.

        Parameters
        ----------
            learning_rate_info : dict
                Here some example:
                "learning_rate": {
                        "type": "ExponentialDecay",
                        "params": {
                            "lr": 1e-2,
                            "decay_steps": 100,
                            ...
                        },
                    }
                For more examples, visit example_of_builders.json in this folder.

        Returns
        -------
            learning_rate : tensorflow learning rate
                Built learning rate.
            global_step : tf.Variable
                 Optional Variable to increment by one after the variables have been updated.
                 None if `learning_rate_info` is not dict.
        """
        if type(learning_rate_info) is dict:
            opt_type = learning_rate_info[LearningRateBuilder.TYPE_FIELD]
            params = learning_rate_info[LearningRateBuilder.PARAMS]
            global_step = tf.Variable(0, trainable=False)
            build_dict = {
                LearningRateBuilder.EXPONENTIAL_DECAY: LearningRateBuilder.__exponential_decay_learning_rate,
                LearningRateBuilder.COSINE_DECAY: LearningRateBuilder.__cosine_decay_learning_rate,
                LearningRateBuilder.COSINE_RESTARTS_DECAY: LearningRateBuilder.__cosine_restarts_decay_learning_rate,
                LearningRateBuilder.INVERSE_TIME_DECAY: LearningRateBuilder.__inverse_time_decay_learning_rate,
                LearningRateBuilder.LINEAR_COSINE_DECAY: LearningRateBuilder.__linear_cosine_decay_learning_rate,
                LearningRateBuilder.NATURAL_EXP_DECAY: LearningRateBuilder.__natural_exp_decay_learning_rate,
                LearningRateBuilder.NOISE_LINEAR_COSINE_DECAY: LearningRateBuilder.__noisy_linear_cosine_decay_learning_rate,
                LearningRateBuilder.PIECEWISE_CONSTANT_DECAY: LearningRateBuilder.__piecewise_constant_decay_learning_rate,
                LearningRateBuilder.POLYNOMIAL_DECAY: LearningRateBuilder.__polynomial_decay_learning_rate,
            }
            return build_dict[opt_type](params, global_step), global_step
        else:
            return learning_rate_info, None

    @staticmethod
    def __exponential_decay_learning_rate(params, global_step):
        lr = params[LearningRateBuilder.LEARNING_RATE]
        decay_steps = params[LearningRateBuilder.DECAY_STEPS]
        decay_rate = params[LearningRateBuilder.DECAY_RATE]
        staircase = params[LearningRateBuilder.STAIRCASE]
        name = params[LearningRateBuilder.NAME]
        return tf.train.exponential_decay(
            learning_rate=lr,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name=name
        )

    @staticmethod
    def __cosine_decay_learning_rate(params, global_step):
        lr = params[LearningRateBuilder.LEARNING_RATE]
        decay_steps = params[LearningRateBuilder.DECAY_STEPS]
        alpha = params[LearningRateBuilder.ALPHA]
        name = params[LearningRateBuilder.NAME]
        return tf.train.cosine_decay(
            learning_rate=lr,
            global_step=global_step,
            decay_steps=decay_steps,
            alpha=alpha,
            name=name
        )

    @staticmethod
    def __cosine_restarts_decay_learning_rate(params, global_step):
        lr = params[LearningRateBuilder.LEARNING_RATE]
        decay_steps = params[LearningRateBuilder.DECAY_STEPS]
        t_mul = params[LearningRateBuilder.T_MUL]
        m_mul = params[LearningRateBuilder.M_MUL]
        alpha = params[LearningRateBuilder.ALPHA]
        name = params[LearningRateBuilder.NAME]
        return tf.train.cosine_decay_restarts(
            learning_rate=lr,
            global_step=global_step,
            first_decay_steps=decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha,
            name=name
        )

    @staticmethod
    def __inverse_time_decay_learning_rate(params, global_step):
        lr = params[LearningRateBuilder.LEARNING_RATE]
        decay_steps = params[LearningRateBuilder.DECAY_STEPS]
        decay_rate = params[LearningRateBuilder.DECAY_RATE]
        staircase = params[LearningRateBuilder.STAIRCASE]
        name = params[LearningRateBuilder.NAME]
        return tf.train.inverse_time_decay(
            learning_rate=lr,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name=name
        )

    @staticmethod
    def __linear_cosine_decay_learning_rate(params, global_step):
        lr = params[LearningRateBuilder.LEARNING_RATE]
        decay_steps = params[LearningRateBuilder.DECAY_STEPS]
        num_periods = params[LearningRateBuilder.NUM_PERIODS]
        alpha = params[LearningRateBuilder.ALPHA]
        beta = params[LearningRateBuilder.BETA]
        name = params[LearningRateBuilder.NAME]
        return tf.train.linear_cosine_decay(
            learning_rate=lr,
            global_step=global_step,
            decay_steps=decay_steps,
            num_periods=num_periods,
            alpha=alpha,
            beta=beta,
            name=name
        )

    @staticmethod
    def __natural_exp_decay_learning_rate(params, global_step):
        lr = params[LearningRateBuilder.LEARNING_RATE]
        decay_steps = params[LearningRateBuilder.DECAY_STEPS]
        decay_rate = params[LearningRateBuilder.DECAY_RATE]
        staircase = params[LearningRateBuilder.STAIRCASE]
        name = params[LearningRateBuilder.NAME]
        return tf.train.natural_exp_decay(
            learning_rate=lr,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name=name
        )

    @staticmethod
    def __noisy_linear_cosine_decay_learning_rate(params, global_step):
        lr = params[LearningRateBuilder.LEARNING_RATE]
        decay_steps = params[LearningRateBuilder.DECAY_STEPS]
        initial_variance = params[LearningRateBuilder.INITIAL_VARIANCE]
        variance_decay = params[LearningRateBuilder.VARIANCE_DECAY]
        num_periods = params[LearningRateBuilder.NUM_PERIODS]
        alpha = params[LearningRateBuilder.ALPHA]
        beta = params[LearningRateBuilder.BETA]
        name = params[LearningRateBuilder.NAME]
        return tf.train.noisy_linear_cosine_decay(
            learning_rate=lr,
            global_step=global_step,
            decay_steps=decay_steps,
            initial_variance=initial_variance,
            variance_decay=variance_decay,
            num_periods=num_periods,
            alpha=alpha,
            beta=beta,
            name=name
        )

    @staticmethod
    def __piecewise_constant_decay_learning_rate(params, global_step):
        boundaries = params[LearningRateBuilder.BOUNDARIES]
        values = params[LearningRateBuilder.VALUES]
        name = params[LearningRateBuilder.NAME]
        return tf.train.piecewise_constant_decay(
            x=global_step,
            boundaries=boundaries,
            values=values,
            name=name
        )

    @staticmethod
    def __polynomial_decay_learning_rate(params, global_step):
        lr = params[LearningRateBuilder.LEARNING_RATE]
        decay_steps = params[LearningRateBuilder.DECAY_STEPS]
        end_learning_rate = params[LearningRateBuilder.END_LEARNING_RATE]
        power = params[LearningRateBuilder.POWER]
        cycle = params[LearningRateBuilder.CYCLE]
        name = params[LearningRateBuilder.NAME]
        return tf.train.polynomial_decay(
            learning_rate=lr,
            global_step=global_step,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate,
            power=power,
            cycle=cycle,
            name=name
        )

