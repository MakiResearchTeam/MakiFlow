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
from makiflow.trainers.utils.learning_rate_builder import LearningRateBuilder

"""
{
    "type": "MomentumOptimizer",
    "params": {
        "lr": ..
        "momentum": ..
    }
}
"""

class OptimizerBuilder:

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
        opt_type = optimizer_info['type']
        params = optimizer_info['params']
        build_dict = {
            'MomentumOptimizer': OptimizerBuilder.__momentum_optimizer,
            'AdamOptimizer': OptimizerBuilder.__adam_optimizer,
            'RMSPropOptimizer': OptimizerBuilder.__rmsprop_optimizer,
            'GradientDescentOptimizer': OptimizerBuilder.__gradient_descent_optimizer,
            'AdadeltaOptimizer': OptimizerBuilder.__adadelta_optimizer,
            'AdagradOptimizer': OptimizerBuilder.__adagrad_optimizer
        }
        return build_dict[opt_type](params)

    @staticmethod
    def __momentum_optimizer(params):
        lr, global_step = LearningRateBuilder.build_learning_rate(params['learning_rate'])
        momentum = params['momentum']
        use_locking = params['use_locking']
        use_nesterov = params['use_nesterov']
        name = params['name']
        return tf.train.MomentumOptimizer(
            learning_rate=lr,
            momentum=momentum,
            use_locking=use_locking,
            use_nesterov=use_nesterov,
            name=name
        ), global_step

    @staticmethod
    def __adam_optimizer(params):
        lr, global_step = LearningRateBuilder.build_learning_rate(params['learning_rate'])
        beta1 = params['beta1']
        beta2 = params['beta2']
        epsilon = params['epsilon']
        use_locking = params['use_locking']
        name = params['name']
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
        lr, global_step = LearningRateBuilder.build_learning_rate(params['learning_rate'])
        decay = params['decay']
        momentum = params['momentum']
        epsilon = params['epsilon']
        use_locking = params['use_locking']
        centered = params['centered']
        name = params['name']
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
        lr, global_step = LearningRateBuilder.build_learning_rate(params['learning_rate'])
        use_locking = params['use_locking']
        name = params['name']
        return tf.train.GradientDescentOptimizer(
            learning_rate=lr,
            use_locking=use_locking,
            name=name
        ), global_step

    @staticmethod
    def __adadelta_optimizer(params):
        lr, global_step = LearningRateBuilder.build_learning_rate(params['learning_rate'])
        rho = params['rho']
        epsilon = params['epsilon']
        use_locking = params['use_locking']
        name = params['name']
        return tf.train.AdadeltaOptimizer(
            learning_rate=lr,
            rho=rho,
            epsilon=epsilon,
            use_locking=use_locking,
            name=name
        ), global_step

    @staticmethod
    def __adagrad_optimizer(params):
        lr, global_step = LearningRateBuilder.build_learning_rate(params['learning_rate'])
        initial_accumulator_value = params['initial_accumulator_value']
        use_locking = params['use_locking']
        name = params['name']
        return tf.train.AdagradOptimizer(
            learning_rate=lr,
            initial_accumulator_value=initial_accumulator_value,
            use_locking=use_locking,
            name=name
        ), global_step
