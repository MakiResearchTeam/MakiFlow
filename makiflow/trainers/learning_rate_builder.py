import tensorflow as tf

"""
{
    "type": "ExponentialDecay",
    "params": {
        "lr": ..
        "decay_steps": ..
    }
}
"""


class LearningRateBuilder:

    @staticmethod
    def build_learning_rate(learning_rate_info):
        """
        Build learning rate with curtain params.

        Parameters
        ----------
            learning_rate_info : dict
                Here some example:
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
            learning_rate : tensorflow learning rate
                Built learning rate.
            global_step : tf.Variable
                 Optional Variable to increment by one after the variables have been updated.
                 None if `learning_rate_info` is not dict.
        """
        if type(learning_rate_info) is type(dict):
            opt_type = learning_rate_info['type']
            params = learning_rate_info['params']
            global_step  = tf.Variable(0, trainable=False)
            build_dict = {
                'ExponentialDecay': LearningRateBuilder.__exponential_decay_learning_rate,
                'CosineDecay': LearningRateBuilder.__cosine_decay_learning_rate,
                'CosineRestartsDecay': LearningRateBuilder.__cosine_restarts_decay_learning_rate,
                'InverseTimeDecay': LearningRateBuilder.__inverse_time_decay_learning_rate,
                'LinearCosineDecay': LearningRateBuilder.__linear_cosine_decay_learning_rate,
                'NaturalExpDecay': LearningRateBuilder.__natural_exp_decay_learning_rate,
                'NoiseLinearCosineDecay': LearningRateBuilder.__noisy_linear_cosine_decay_learning_rate,
                'PiecewiseConstantDecay': LearningRateBuilder.__piecewise_constant_decay_learning_rate,
                'PolynomialDecay': LearningRateBuilder.__polynomial_decay_learning_rate,
            }
            return build_dict[opt_type](params, global_step), global_step
        else:
            return learning_rate_info, None

    @staticmethod
    def __exponential_decay_learning_rate(params, global_step):
        lr = params['lr']
        decay_steps = params['decay_steps']
        decay_rate = params['decay_rate']
        staircase = params['staircase']
        name = params['name']
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
        lr = params['lr']
        decay_steps = params['decay_steps']
        alpha = params['alpha']
        name = params['name']
        return tf.train.cosine_decay(
            learning_rate=lr,
            global_step=global_step,
            decay_steps=decay_steps,
            alpha=alpha,
            name=name
        )

    @staticmethod
    def __cosine_restarts_decay_learning_rate(params, global_step):
        lr = params['lr']
        decay_steps = params['decay_steps']
        t_mul = params['t_mul']
        m_mul = params['m_mul']
        alpha = params['alpha']
        name = params['name']
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
        lr = params['lr']
        decay_steps = params['decay_steps']
        decay_rate = params['decay_rate']
        staircase = params['staircase']
        name = params['name']
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
        lr = params['lr']
        decay_steps = params['decay_steps']
        num_periods = params['num_periods']
        alpha = params['alpha']
        beta = params['beta']
        name = params['name']
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
        lr = params['lr']
        decay_steps = params['decay_steps']
        decay_rate = params['decay_rate']
        staircase = params['staircase']
        name = params['name']
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
        lr = params['lr']
        decay_steps = params['decay_steps']
        initial_variance = params['initial_variance']
        variance_decay = params['variance_decay']
        num_periods = params['num_periods']
        alpha = params['alpha']
        beta = params['beta']
        name = params['name']
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
        boundaries = params['boundaries']
        values = params['values']
        name = params['name']
        return tf.train.piecewise_constant_decay(
            x=global_step,
            boundaries=boundaries,
            values=values,
            name=name
        )

    @staticmethod
    def __polynomial_decay_learning_rate(params, global_step):
        lr = params['lr']
        decay_steps = params['decay_steps']
        end_learning_rate = params['end_learning_rate']
        power = params['power']
        cycle = params['cycle']
        name = params['name']
        return tf.train.polynomial_decay(
            learning_rate=lr,
            global_step=global_step,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate,
            power=power,
            cycle=cycle,
            name=name
        )

