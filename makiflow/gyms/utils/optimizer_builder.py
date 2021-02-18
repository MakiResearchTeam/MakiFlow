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
from makiflow.gyms import LearningRateBuilder

"""
{
    "type": "MomentumOptimizer",
    "params": {
        "learning_rate": ..
        "momentum": ..
    }
}
"""


class OptimizerBuilder:

    TYPE_FIELD = 'type'
    PARAMS = 'params'
    LEARNING_RATE = 'learning_rate'
    NAME = 'name'
    USE_LOCKING = 'use_locking'
    GRADIENT_DESCENT_OPTIMIZER = 'GradientDescentOptimizer'

    # For Momentum
    MOMENTUM_OPTIMIZER = 'MomentumOptimizer'
    MOMENTUM = 'momentum'
    USE_NESTEROV = 'use_nesterov'

    # For Adam
    ADAM_OPTIMIZER = 'AdamOptimizer'
    BETA1 = 'beta1'
    BETA2 = 'beta2'
    EPSILON = 'epsilon'

    # For RMSProp
    RMSPROP_OPTIMIZER = 'RMSPropOptimizer'
    DECAY = 'decay'
    CENTERED = 'centered'

    # For AdaDelta
    ADADELTA_OPTIMIZER = 'AdadeltaOptimizer'
    RHO = 'rho'

    # AdaGrad
    ADAGRAD_OPTIMIZER = 'AdagradOptimizer'
    INITIAL_ACCUMULATOR_VALUE = 'initial_accumulator_value'

    @staticmethod
    def build_optimizer(optimizer_info):
        """
        Build optimizer with certain params.

        Parameters
        ----------
            optimizer_info : dict
                Here some example:
                {
                    "type": "MomentumOptimizer",
                    "params": {
                        "lr": ..
                        "momentum": ..
                    }
                }
                Where `lr` can be, for example:
                "lr": {
                        "type": "ExponentialDecay",
                        "params": {
                            "lr": ..
                            "decay_steps": ..
                        }
                    }
                For more examples, visit example_of_builders.json in this folder.

        Returns
        -------
            optimizer : tensorflow optimizer
                Built optimizer.
            global_step : tf.Variable
                 Optional Variable to increment by one after the variables have been updated.
        """
        opt_type = optimizer_info[OptimizerBuilder.TYPE_FIELD]
        params = optimizer_info[OptimizerBuilder.PARAMS]
        build_dict = {
            OptimizerBuilder.MOMENTUM_OPTIMIZER: OptimizerBuilder.__momentum_optimizer,
            OptimizerBuilder.ADAM_OPTIMIZER: OptimizerBuilder.__adam_optimizer,
            OptimizerBuilder.RMSPROP_OPTIMIZER: OptimizerBuilder.__rmsprop_optimizer,
            OptimizerBuilder.GRADIENT_DESCENT_OPTIMIZER: OptimizerBuilder.__gradient_descent_optimizer,
            OptimizerBuilder.ADADELTA_OPTIMIZER: OptimizerBuilder.__adadelta_optimizer,
            OptimizerBuilder.ADAGRAD_OPTIMIZER: OptimizerBuilder.__adagrad_optimizer
        }
        return build_dict[opt_type](params)

    @staticmethod
    def __momentum_optimizer(params):
        lr, global_step = LearningRateBuilder.build_learning_rate(params[OptimizerBuilder.LEARNING_RATE])
        momentum = params[OptimizerBuilder.MOMENTUM]
        use_locking = params[OptimizerBuilder.USE_LOCKING]
        use_nesterov = params[OptimizerBuilder.USE_NESTEROV]
        name = params[OptimizerBuilder.NAME]
        return tf.train.MomentumOptimizer(
            learning_rate=lr,
            momentum=momentum,
            use_locking=use_locking,
            use_nesterov=use_nesterov,
            name=name
        ), global_step

    @staticmethod
    def __adam_optimizer(params):
        lr, global_step = LearningRateBuilder.build_learning_rate(params[OptimizerBuilder.LEARNING_RATE])
        beta1 = params[OptimizerBuilder.BETA1]
        beta2 = params[OptimizerBuilder.BETA2]
        epsilon = params[OptimizerBuilder.EPSILON]
        use_locking = params[OptimizerBuilder.USE_LOCKING]
        name = params[OptimizerBuilder.NAME]
        return tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            use_locking=use_locking,
            name=name
        ), global_step

    @staticmethod
    def __rmsprop_optimizer(params):
        lr, global_step = LearningRateBuilder.build_learning_rate(params[OptimizerBuilder.LEARNING_RATE])
        decay = params[OptimizerBuilder.DECAY]
        momentum = params[OptimizerBuilder.MOMENTUM]
        epsilon = params[OptimizerBuilder.EPSILON]
        use_locking = params[OptimizerBuilder.USE_LOCKING]
        centered = params[OptimizerBuilder.CENTERED]
        name = params[OptimizerBuilder.NAME]
        return tf.train.RMSPropOptimizer(
            learning_rate=lr,
            decay=decay,
            momentum=momentum,
            epsilon=epsilon,
            use_locking=use_locking,
            centered=centered,
            name=name
        ), global_step

    @staticmethod
    def __gradient_descent_optimizer(params):
        lr, global_step = LearningRateBuilder.build_learning_rate(params[OptimizerBuilder.LEARNING_RATE])
        use_locking = params[OptimizerBuilder.USE_LOCKING]
        name = params[OptimizerBuilder.NAME]
        return tf.train.GradientDescentOptimizer(
            learning_rate=lr,
            use_locking=use_locking,
            name=name
        ), global_step

    @staticmethod
    def __adadelta_optimizer(params):
        lr, global_step = LearningRateBuilder.build_learning_rate(params[OptimizerBuilder.LEARNING_RATE])
        rho = params[OptimizerBuilder.RHO]
        epsilon = params[OptimizerBuilder.EPSILON]
        use_locking = params[OptimizerBuilder.USE_LOCKING]
        name = params[OptimizerBuilder.NAME]
        return tf.train.AdadeltaOptimizer(
            learning_rate=lr,
            rho=rho,
            epsilon=epsilon,
            use_locking=use_locking,
            name=name
        ), global_step

    @staticmethod
    def __adagrad_optimizer(params):
        lr, global_step = LearningRateBuilder.build_learning_rate(params[OptimizerBuilder.LEARNING_RATE])
        initial_accumulator_value = params[OptimizerBuilder.INITIAL_ACCUMULATOR_VALUE]
        use_locking = params[OptimizerBuilder.USE_LOCKING]
        name = params[OptimizerBuilder.NAME]
        return tf.train.AdagradOptimizer(
            learning_rate=lr,
            initial_accumulator_value=initial_accumulator_value,
            use_locking=use_locking,
            name=name
        ), global_step

