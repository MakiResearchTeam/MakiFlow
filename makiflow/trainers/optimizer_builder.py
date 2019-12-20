import tensorflow as tf
from .learning_rate_builder import LearningRateBuilder

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
        lr = LearningRateBuilder.build_learning_rate(params['learning_rate'])
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
        )

    @staticmethod
    def __adam_optimizer(params):
        lr = LearningRateBuilder.build_learning_rate(params['learning_rate'])
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
        )

    @staticmethod
    def __rmsprop_optimizer(params):
        lr = LearningRateBuilder.build_learning_rate(params['learning_rate'])
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
        )

    @staticmethod
    def __gradient_descent_optimizer(params):
        lr = LearningRateBuilder.build_learning_rate(params['learning_rate'])
        use_locking = params['use_locking']
        name = params['name']
        return tf.train.GradientDescentOptimizer(
            learning_rate=lr,
            use_locking=use_locking,
            name=name
        )

    @staticmethod
    def __adadelta_optimizer(params):
        lr = LearningRateBuilder.build_learning_rate(params['learning_rate'])
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
        )

    @staticmethod
    def __adagrad_optimizer(params):
        lr = LearningRateBuilder.build_learning_rate(params['learning_rate'])
        initial_accumulator_value = params['initial_accumulator_value']
        use_locking = params['use_locking']
        name = params['name']
        return tf.train.AdagradOptimizer(
            learning_rate=lr,
            initial_accumulator_value=initial_accumulator_value,
            use_locking=use_locking,
            name=name
        )

