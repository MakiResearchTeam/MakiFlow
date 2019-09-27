import tensorflow as tf
import numpy as np


def vec_len(vec):
    vec2 = vec * vec
    vec_sum = tf.reduce_sum(vec2)
    return tf.sqrt(vec_sum)


def distance(vec1, vec2):
    diff = vec1 - vec2
    return vec_len(diff)


def normalize(vec):
    vec_l = vec_len(vec)
    return vec / vec_l


class GDBalancer:
    def __init__(self, optimizer, to_balance, objective='alpha', min_amount=5.0, max_amount=10000.0,
                 initial_amount=1000):
        # for GradientDescentOptimizer optimal lr=1000
        # for AdamOptimizer optimal lr=1

        self.to_balance = to_balance
        self.vecs = tf.constant(to_balance.T, dtype=tf.float32)
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.dest_v = tf.placeholder(dtype=tf.float32, shape=[to_balance.shape[1], 1])
        # Setup the algorithm
        self.sess = None
        self.reset(initial_amount, optimizer, objective)

    def optimize(self, destination_v, iterations=10, print_period=5):
        for i in range(iterations):
            _, d = self.sess.run([self.train_op, self.objective],
                                 feed_dict={self.dest_v: destination_v}
                                 )
            if i % print_period == 0:
                print(d)
                print(self.get_percentage())

    def get_percentage(self):
        percentage = self.get_scaled_vecs() / self.get_vec_num()
        return np.round(percentage, decimals=2) * 100

    def get_weights(self):
        return self.sess.run(self.w)

    def get_scaled_vecs(self):
        return self.sess.run(self.scaled_vecs)

    def get_vec_num(self):
        return self.sess.run(tf.reduce_sum(self.w))

    def reset(self, initial_amount, optimizer, objective):
        if self.sess is not None:
            self.sess.close()

        self.optimizer = optimizer
        w = np.ones((len(self.to_balance), 1)) * initial_amount
        self.w = tf.Variable(
            w,
            trainable=True,
            # Clip the value of the weights in the interval [min_amount, max_amount]
            constraint=lambda x: tf.clip_by_value(x, self.min_amount, self.max_amount),
            dtype=tf.float32
        )
        self.scaled_vecs = tf.matmul(self.vecs, self.w)

        self.build_objective(objective)

        self.train_op = self.optimizer.minimize(self.objective, var_list=[self.w])
        self.sess = tf.Session()
        # Initialize all the variables
        self.sess.run(tf.variables_initializer(self.optimizer.variables()))
        self.sess.run(tf.variables_initializer([self.w]))

    def build_objective(self, objective):
        if objective == 'alpha':
            norm_vecs = self.scaled_vecs / tf.reduce_sum(self.w)
            self.objective = distance(norm_vecs, self.dest_v)
        elif objective == 'geo':
            norm_vecs = normalize(self.scaled_vecs)
            self.objective = distance(norm_vecs, self.dest_v)
        else:
            print('Unknowm objective. Call `reset` with the correct one.')
